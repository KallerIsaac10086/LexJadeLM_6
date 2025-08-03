# trainer_LexJade6/train_pretrain.py
import os
import sys
__package__ = "trainer"
# 确保 Python 能找到 model 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
# 注意：LexJadeLM6 使用 transformers 的 AutoTokenizer
from transformers import AutoTokenizer 
# 从 model_LexJadeLM6.py 导入配置类和模型类
# 确保导入的是类，而不是实例
from model.model_LexJadeLM6 import (
    LexJade6TinyConfig, LexJade6FlashConfig, LexJade6CubeConfig, LexJade6LargeConfig,
    LexJade6ExtremeConfig, LexJade6UltraConfig, LexJade6ExtremeMConfig, LexJade6UltraMConfig,
    LexJadeForCausalLM
)
from dataset.lm_dataset import PretrainDataset # 确保这个路径正确
warnings.filterwarnings('ignore')
# --- 模型配置映射 (在训练脚本中直接定义，避免导入实例) ---
# 存储配置类本身，而不是实例
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
    loss_fct = nn.CrossEntropyLoss(reduction='none')
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
            # 修改这里：将学习率格式化为10位小数，而不是科学计数法
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
            ckp = f'{args.save_dir}/{model_name}{moe_suffix}_epoch{epoch+1:03d}_step{step+1:05d}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 为了节省空间，可以考虑保存为半精度
            # state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            Logger(f"Checkpoint saved to {ckp}")
            model.train() # 保存后恢复训练模式
def init_model(lm_config, tokenizer_path):
    # 使用 AutoTokenizer 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # 确保 pad_token_id 设置
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 # 或者使用 tokenizer.eos_token_id
    # 使用我们新定义的模型类
    model = LexJadeForCausalLM(config=lm_config).to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'{MODEL_CONFIG_MAP[args.model_choice][0]} 可训练总参数量：{trainable_params / 1e6:.3f} 百万')
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
def select_model():
    """在运行时让用户选择模型"""
    print("\n--- 请选择要预训练的 LexJadeLM6 模型 ---")
    for key, (name, _) in MODEL_CONFIG_MAP.items():
        print(f"{key}. {name}")
    print("------------------------------------------")
    while True:
        choice = input("请输入模型编号 (1-8): ").strip()
        if choice in MODEL_CONFIG_MAP:
            return choice
        else:
            print("无效输入，请输入 1 到 8 之间的数字。")
# torchrun --nproc_per_node 2 trainer_LexJade6/train_pretrain.py --model_choice 6 --epochs 3 --batch_size 8 --accumulation_steps 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexJadeLM6 Pretraining")
    # 新增参数：模型选择 (可以在命令行指定，也可以运行时选择)
    parser.add_argument("--model_choice", type=str, default=None, help="Model choice (1-8), if not provided, a menu will appear.")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. If None, uses the default from the model config or a preset.")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu") # 默认使用第一个可用GPU
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="LexJadeLM6-Pretrain")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    # max_seq_len 现在从配置中获取，但如果命令行指定了，则使用命令行的
    parser.add_argument("--max_seq_len", type=int, default=384, help="Max sequence length. If None, uses the default from the model config.")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    # 分词器和模型文件路径
    parser.add_argument("--tokenizer_path", type=str, default="./model/tokenizer_LexJadeLM6-Large")
    # 新增学习率调度参数
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total steps for learning rate warmup")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of initial learning rate")
    args = parser.parse_args()
    # --- 模型选择逻辑 ---
    if args.model_choice is None:
        args.model_choice = select_model()
    elif args.model_choice not in MODEL_CONFIG_MAP:
        print(f"错误：命令行参数 --model_choice '{args.model_choice}' 无效。")
        print("有效选项为 1 到 8。")
        sys.exit(1)
    # 获取选定的配置类
    model_name, config_class = MODEL_CONFIG_MAP[args.model_choice]
    print(f"\n已选择模型: {model_name}")
    # 实例化配置对象
    # 这会使用配置类中定义的默认值
    lm_config = config_class()
    # --- 处理 max_seq_len 和 batch_size ---
    # 如果命令行没有指定，则从配置中获取
    if args.max_seq_len is None:
        max_seq_len = lm_config.max_position_embeddings
        print(f"使用模型配置的 max_seq_len: {max_seq_len}")
    else:
        max_seq_len = args.max_seq_len
        print(f"使用命令行指定的 max_seq_len: {max_seq_len}")
        # 注意：通常不直接修改 config 对象的 max_position_embeddings，
        # 除非你想让它影响模型初始化。这里我们只用它来创建数据集。
    if args.batch_size is None:
        # 为每个模型设置一个默认的合理 batch_size
        # 这些值需要根据你的GPU内存调整
        default_batch_sizes = {
            "1": 16, "2": 16, "3": 8, "4": 8,
            "5": 4, "6": 2, "7": 4, "8": 2
        }
        args.batch_size = default_batch_sizes.get(args.model_choice, 4) # 默认4
        print(f"使用默认 batch_size for {model_name}: {args.batch_size}")
    else:
        print(f"使用命令行指定的 batch_size: {args.batch_size}")
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 更新 wandb 运行名称以包含模型信息
    args.wandb_run_name = f"{model_name}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
    # 设置自动混合精度上下文
    if args.dtype == "float16":
        ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    elif args.dtype == "bfloat16":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else: # float32
        ctx = nullcontext()
    # DDP 初始化
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    # 如果使用 DDP
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
    # Wandb 初始化
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        wandb = None
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.tokenizer_path)
    # 初始化数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None # 启用 shuffle
    # 注意：如果使用 DDP，shuffle 应在 sampler 中处理
    shuffle = False if ddp else True
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False, # 通常预训练设为 False
        shuffle=shuffle,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    # 初始化梯度缩放器和优化器
    # 根据 dtype 设置 GradScaler 的 enabled 状态
    scaler_enabled = args.dtype in ['float16', 'bfloat16']
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 如果使用 DDP，包装模型
    if ddp:
        # model._ddp_params_and_buffers_to_ignore = {"pos_cis"} # RoPE buffers 通常会自动处理
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    # 训练循环
    iter_per_epoch = len(train_loader)
    Logger(f"开始预训练 {model_name}...")
    Logger(f"总 Epochs: {args.epochs}, 每 Epoch Steps: {iter_per_epoch}, Batch Size: {args.batch_size}, "
           f"Accumulation Steps: {args.accumulation_steps}, Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    Logger(f"Learning Rate: {args.learning_rate}, Sequence Length: {max_seq_len}, Data Path: {args.data_path}")
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
        final_ckp = f'{args.save_dir}/final_{model_name}{moe_suffix}.pth'
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        # state_dict = {k: v.half() for k, v in state_dict.items()} # 可选：半精度保存
        torch.save(state_dict, final_ckp)
        Logger(f"\n最终模型已保存至: {final_ckp}")
        Logger("预训练完成。")