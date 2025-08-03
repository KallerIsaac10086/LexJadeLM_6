# trainer_LexJade6/train_sft.py
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
# 从 model_LexJadeLM6.py 导入配置类和模型类
from model.model_LexJadeLM6 import (
    LexJade6TinyConfig, LexJade6FlashConfig, LexJade6CubeConfig, LexJade6LargeConfig,
    LexJade6ExtremeConfig, LexJade6UltraConfig, LexJade6ExtremeMConfig, LexJade6UltraMConfig,
    LexJadeForCausalLM
)
from dataset.lm_dataset import SFTDataset  # 确保路径正确

warnings.filterwarnings('ignore')

# --- 模型配置映射 ---
MODEL_CONFIG_MAP = {
    "1": ("LexJade6-Tiny", LexJade6TinyConfig),
    "2": ("LexJade6-Flash", LexJade6FlashConfig),
    "3": ("LexJade6-Cube", LexJade6CubeConfig),
    "4": ("LexJade6-Large", LexJade6LargeConfig),
    "5": ("LexJade6-Extreme", LexJade6ExtremeConfig),
    "6": ("LexJade6-Ultra", LexJade6UltraConfig),
    "7": ("LexJade6-Extreme-M", LexJade6ExtremeMConfig),
    "8": ("LexJade6-Ultra-M", LexJade6UltraMConfig),
}

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr, warmup_ratio=0.05, min_lr_ratio=0.1):
    """
    优化的学习率退火函数，包含预热阶段和最小学习率限制
    
    参数:
    current_step: 当前训练步数
    total_steps: 总训练步数
    lr: 初始学习率
    warmup_ratio: 预热步数占总步数的比例 (默认为5%)
    min_lr_ratio: 最小学习率与初始学习率的比例 (默认为10%)
    
    返回:
    当前学习率
    """
    # 计算预热步数
    warmup_steps = int(total_steps * warmup_ratio)
    
    # 预热阶段: 从0线性增加到初始学习率
    if current_step < warmup_steps:
        return lr * (current_step + 1) / warmup_steps
    
    # 退火阶段: 余弦退火衰减到最小学习率
    decay_ratio = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    min_lr = lr * min_lr_ratio
    return min_lr + coeff * (lr - min_lr)

def train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, iter_per_epoch, lm_config):
    start_time = time.time()
    # 确保模型处于训练模式
    model.train() 
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        current_step = epoch * iter_per_epoch + step
        # 使用优化后的学习率调度函数
        lr = get_lr(current_step, args.epochs * iter_per_epoch, args.learning_rate,
                   args.warmup_ratio, args.min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with ctx:
            # 调用模型的 forward 方法
            outputs = model(input_ids=X, labels=Y) 
            loss = outputs.loss # 模型内部已计算包含 aux_loss 的总损失
            # 应用梯度累积
            loss = loss / args.accumulation_steps 
        # 反向传播
        scaler.scale(loss).backward()
        # 梯度累积更新
        if (step + 1) % args.accumulation_steps == 0:
            if args.device.startswith("cuda"): # 仅在 CUDA 上进行梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps # 还原累积前的损失
            # 计算剩余时间估计 (以分钟为单位)
            if step > 0 and iter_per_epoch > 0:
                time_per_step = spend_time / (step + 1)
                remaining_steps = iter_per_epoch - step - 1 + (args.epochs - epoch - 1) * iter_per_epoch
                eta_seconds = time_per_step * remaining_steps
                eta_minutes = eta_seconds / 60.0 # 转换为分钟
            else:
                eta_minutes = float('inf') # 或者设置为一个大数，表示未知
            Logger(
                f'Epoch:[{epoch + 1:03d}/{args.epochs:03d}]({step:04d}/{iter_per_epoch:04d}) '
                f'loss:{current_loss:.3f} lr:{lr:.10f} ETA:{eta_minutes:.1f}min'
            )
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": current_loss,
                    "lr": lr,
                    "eta_minutes": eta_minutes if step > 0 and iter_per_epoch > 0 else None
                })
        # 定期保存检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
            model_name = MODEL_CONFIG_MAP[args.model_choice][0] # 获取模型名称
            ckp = f'{args.save_dir}/sft_{model_name}{moe_suffix}_epoch{epoch+1:03d}_step{step+1:05d}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 为了节省空间，可以考虑保存为半精度
            # state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            Logger(f"Checkpoint saved to {ckp}")
            model.train() # 保存后恢复训练模式

def init_model(lm_config, tokenizer_path, model_choice, pretrain_base_dir):
    # 使用 AutoTokenizer 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # 确保 pad_token_id 设置
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 # 或者使用 tokenizer.eos_token_id
    
    # 使用 LexJadeForCausalLM 创建模型
    model = LexJadeForCausalLM(config=lm_config)
    model = model.to(args.device) # 将模型移动到指定设备
    
    # 加载预训练权重
    model_name = MODEL_CONFIG_MAP[model_choice][0]
    moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
    
    # 构建预训练权重目录路径
    pretrain_model_dir = os.path.join(pretrain_base_dir, model_name)
    
    # 尝试加载 "final_*.pth"
    final_ckp_path = os.path.join(pretrain_model_dir, f"final_{model_name}{moe_suffix}.pth")
    # 尝试加载最新的检查点 (按文件名排序，取最后一个)
    all_ckps = []
    if os.path.exists(pretrain_model_dir):
        for f in os.listdir(pretrain_model_dir):
            if f.startswith(model_name) and f.endswith('.pth') and not f.startswith('final_'):
                all_ckps.append(f)
    
    # 确定要加载的权重文件
    ckp_to_load = None
    if os.path.exists(final_ckp_path):
        ckp_to_load = final_ckp_path
        Logger(f"找到最终预训练权重: {final_ckp_path}")
    elif all_ckps:
        # 按文件名排序，取最新的
        all_ckps.sort()
        latest_ckp = all_ckps[-1]
        ckp_to_load = os.path.join(pretrain_model_dir, latest_ckp)
        Logger(f"找到最新预训练检查点: {ckp_to_load}")
    else:
        Logger(f"警告: 未在 {pretrain_model_dir} 找到预训练权重，将从随机初始化开始SFT训练")
    
    # 加载权重
    if ckp_to_load and os.path.exists(ckp_to_load):
        try:
            # map_location 确保权重加载到正确的设备上
            state_dict = torch.load(ckp_to_load, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
            Logger(f"成功加载预训练权重: {ckp_to_load}")
        except Exception as e:
            Logger(f"警告: 加载预训练权重失败 {ckp_to_load}: {e}")
    else:
        Logger("警告: 没有找到可用的预训练权重，将从随机初始化开始SFT训练")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'{model_name} 可训练总参数量：{trainable_params / 1e6:.3f} 百万')
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def select_model(pretrain_base_dir):
    """在运行时让用户选择模型"""
    print("\n--- 请选择要进行SFT的 LexJadeLM6 模型 ---")
    available_models = []
    
    # 检查预训练目录中的模型
    if not os.path.exists(pretrain_base_dir):
        print(f"错误: 预训练根目录 {pretrain_base_dir} 不存在!")
        sys.exit(1)
        
    # 遍历 MODEL_CONFIG_MAP 中的模型
    for key, (name, _) in MODEL_CONFIG_MAP.items():
        model_dir = os.path.join(pretrain_base_dir, name)
        final_ckp = os.path.join(model_dir, f"final_{name}.pth")
        moe_final_ckp = os.path.join(model_dir, f"final_{name}_moe.pth")
        
        has_weights = False
        ckp_info = ""
        
        if os.path.exists(final_ckp):
            has_weights = True
            ckp_info = f"(最终权重: {final_ckp})"
        elif os.path.exists(moe_final_ckp):
            has_weights = True
            ckp_info = f"(最终权重: {moe_final_ckp})"
        elif os.path.exists(model_dir) and os.path.isdir(model_dir):
            # 检查是否有检查点文件
            ckps = [f for f in os.listdir(model_dir) if f.startswith(name) and f.endswith('.pth')]
            if ckps:
                has_weights = True
                ckp_info = f"({len(ckps)} 个检查点文件)"
        
        if has_weights:
            available_models.append(key)
            print(f"{key}. {name} {ckp_info}")
        else:
            print(f"{key}. {name} (无预训练权重)")
    
    print("------------------------------------------")
    print("注意: 只有预训练权重存在的模型才能进行SFT训练")
    print("------------------------------------------")
    
    if not available_models:
        print("错误: 在指定目录中未找到任何可用的预训练模型!")
        sys.exit(1)
    
    while True:
        choice = input("请输入模型编号 (例如: 1, 2, 3...) 或输入 'q' 退出: ").strip()
        if choice.lower() == 'q':
            sys.exit(0)
        if choice in available_models:
            return choice
        else:
            available_str = ", ".join(available_models)
            print(f"无效输入，请输入可用模型的编号 ({available_str}) 或 'q' 退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexJadeLM6 SFT")
    # 新增参数：模型选择
    parser.add_argument("--model_choice", type=str, default=None, help="Model choice (1-8), if not provided, a menu will appear.")
    parser.add_argument("--out_dir", type=str, default="./sft_out")
    parser.add_argument("--pretrain_dir", type=str, default="./out", help="Directory containing pre-trained models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # 设备选择：默认cuda，如果没有cuda则使用cpu
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="LexJadeLM6-SFT")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=2000)
    # max_seq_len 现在从配置中获取，但如果命令行指定了，则使用命令行的
    parser.add_argument("--max_seq_len", type=int, default=382, help="Max sequence length. If None, uses the default from the model config.")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/minimind/dataset/sft_mini_51.jsonl")
    # 分词器路径
    parser.add_argument("--tokenizer_path", type=str, default="./model/tokenizer_LexJadeLM6")
    # 新增学习率调度参数
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total steps for learning rate warmup")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of initial learning rate")
    args = parser.parse_args()

    # --- 模型选择逻辑 ---
    if args.model_choice is None:
        args.model_choice = select_model(args.pretrain_dir)
    elif args.model_choice not in MODEL_CONFIG_MAP:
        print(f"错误：命令行参数 --model_choice '{args.model_choice}' 无效。")
        print("有效选项为 1 到 8。")
        sys.exit(1)
    
    # 获取选定的配置类
    model_name, config_class = MODEL_CONFIG_MAP[args.model_choice]
    print(f"\n已选择模型: {model_name}")
    
    # 实例化配置对象
    lm_config = config_class()
    
    # --- 处理 max_seq_len ---
    if args.max_seq_len is None:
        max_seq_len = lm_config.max_position_embeddings
        print(f"使用模型配置的 max_seq_len: {max_seq_len}")
    else:
        max_seq_len = args.max_seq_len
        print(f"使用命令行指定的 max_seq_len: {max_seq_len}")

    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    tokens_per_iter = args.batch_size * max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 更新 wandb 运行名称以包含模型信息
    args.wandb_run_name = f"{model_name}-SFT-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
    
    # 设置自动混合精度上下文
    if args.dtype == "float16":
        ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device_type == "cuda" else nullcontext()
    elif args.dtype == "bfloat16":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    else:  # float32
        ctx = nullcontext()
    
    # DDP 初始化 (仅在 CUDA 环境下支持)
    if args.ddp and device_type != "cuda":
        print("警告: DDP 仅在 CUDA 环境下支持，将禁用 DDP。")
        args.ddp = False
        
    ddp = args.ddp and int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(base_seed)
    
    # 如果使用 DDP
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        if device_type == "cuda":
            torch.cuda.manual_seed(base_seed + rank)
    
    # Wandb 初始化
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        wandb = None
    
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.tokenizer_path, args.model_choice, args.pretrain_dir)
    
    # 初始化数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    shuffle = False if ddp else True
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=(device_type == "cuda"), # 仅在 CUDA 下使用 pin_memory
        drop_last=False,
        shuffle=shuffle,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    
    # 初始化梯度缩放器和优化器
    scaler_enabled = args.dtype in ['float16', 'bfloat16'] and device_type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 如果使用 DDP，包装模型 (仅在 CUDA 环境下)
    if ddp and device_type == "cuda":
        # model._ddp_params_and_buffers_to_ignore = {"pos_cis"} # RoPE buffers 通常会自动处理
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    elif ddp and device_type != "cuda":
        print("警告: DDP 仅在 CUDA 环境下支持，将不使用 DDP。")
        ddp = False
    
    # 训练循环
    iter_per_epoch = len(train_loader)
    Logger(f"开始SFT训练 {model_name}...")
    Logger(f"总 Epochs: {args.epochs}, 每 Epoch Steps: {iter_per_epoch}, Batch Size: {args.batch_size}, "
           f"Accumulation Steps: {args.accumulation_steps}, Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    Logger(f"Learning Rate: {args.learning_rate}, Sequence Length: {max_seq_len}, Data Path: {args.data_path}")
    Logger(f"Device: {args.device}, Data Type: {args.dtype}")
    Logger("-" * 80)
    
    for epoch in range(args.epochs):
        # 如果使用 DDP，设置 epoch 以确保 shuffle 正确工作
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb, model, train_loader, optimizer, scaler, args, iter_per_epoch, lm_config)
    
    # 训练结束，保存最终模型
    if not ddp or dist.get_rank() == 0:
        model.eval()
        moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
        final_ckp = f'{args.save_dir}/final_sft_{model_name}{moe_suffix}.pth'
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        # state_dict = {k: v.half() for k, v in state_dict.items()} # 可选：半精度保存
        torch.save(state_dict, final_ckp)
        Logger(f"\n最终SFT模型已保存至: {final_ckp}")
        Logger("SFT训练完成。")