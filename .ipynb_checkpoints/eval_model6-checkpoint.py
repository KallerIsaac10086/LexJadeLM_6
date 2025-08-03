#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/11/12 15:31
# @Author  : Han
# @File    : eval_model6.py
# @Software: VS Code
# @Desc    : 评估 LexJadeLM6 系列模型的脚本
#            支持多种模型系列 (Tiny/Flash/Cube/Large/Extreme/Ultra/Extreme-M/Ultra-M)
#            支持自动扫描模型、加载 Hugging Face 格式模型、手动指定 .pth 路径
#            支持 Pretrain 和 SFT/Chat 模式
#            支持自动测试和手动输入模式
#            支持历史对话上下文 (仅限 Chat 模式)
#            支持 TextStreamer (已启用)
import os
import sys
import re
import torch
import warnings
from transformers import AutoTokenizer, TextStreamer
# 确保项目根目录在 sys.path 中，以便正确导入自定义模型
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已将项目根目录添加到 sys.path: {project_root}")
# - 导入自定义模型和配置 -
try:
    from model.model_LexJadeLM6 import (
        LexJade6TinyConfig, LexJade6FlashConfig, LexJade6CubeConfig, LexJade6LargeConfig,
        LexJade6ExtremeConfig, LexJade6UltraConfig,
        LexJade6ExtremeMConfig, LexJade6UltraMConfig,
        LexJadeModel, LexJadeForCausalLM
    )
    print("模型模块导入成功。")
except (ImportError, SyntaxError) as e:
    print(f"导入模型模块失败: {e}")
    print("请确保 'model/model_LexJadeLM6.py' 在正确的路径下，并且没有语法错误。")
    exit(1)
# 如果你有 LoRA 工具函数，请确保正确导入
# from model.model_lora import *
warnings.filterwarnings('ignore')
# - LexJadeLM6 模型配置映射 -
MODEL_CONFIG_MAP = {
    "LexJade6-Tiny": LexJade6TinyConfig,
    "LexJade6-Flash": LexJade6FlashConfig,
    "LexJade6-Cube": LexJade6CubeConfig,
    "LexJade6-Large": LexJade6LargeConfig,
    "LexJade6-Extreme": LexJade6ExtremeConfig,
    "LexJade6-Ultra": LexJade6UltraConfig,
    "LexJade6-Extreme-M": LexJade6ExtremeMConfig,
    "LexJade6-Ultra-M": LexJade6UltraMConfig,
}

def get_base_out_dir():
    """获取基础模型输出目录"""
    while True:
        try:
            mode_choice = int(input("\n--- 请选择模型类型 ---\n[0] Pretrain 模型 (out 目录)\n[1] SFT/Chat 模型 (sft_out 目录)\n请输入选项 (0 或 1): "))
            if mode_choice == 0:
                return "./out"
            elif mode_choice == 1:
                return "./sft_out"
            else:
                print("无效输入，请输入 0 或 1。")
        except ValueError:
            print("请输入一个有效的数字。")

def find_available_models(base_out_dir):
    """在指定目录下查找可用的模型检查点"""
    available_models = []
    if not os.path.exists(base_out_dir):
        print(f"警告: 模型输出目录 '{base_out_dir}' 不存在。")
        return available_models
    # 遍历 base_out_dir 下的子目录（每个子目录对应一个模型系列）
    for model_series_dir in os.listdir(base_out_dir):
        series_path = os.path.join(base_out_dir, model_series_dir)
        if os.path.isdir(series_path):
            # 在每个模型系列目录下查找 .pth 文件
            for file in os.listdir(series_path):
                if file.endswith('.pth'):
                    full_path = os.path.join(series_path, file)
                    # 尝试从文件名推断模型类型和模式
                    # 假设文件名格式类似: LexJade6-Flash_epoch001_step01000.pth 或 final_LexJade6-Flash.pth
                    # 或者包含 pretrain / sft 等关键字
                    mode = "Unknown"
                    # 根据目录名判断模式
                    if 'sft' in base_out_dir.lower():
                        mode = "SFT/Chat"
                    else: # 'out' 目录
                        mode = "Pretrain"
                    
                    # 从文件名或目录名中提取模型系列名称
                    model_series = None
                    # 匹配类似 LexJade6-Flash 的模式
                    match = re.search(r'(LexJade6-[A-Za-z0-9\-]+)', file)
                    if match:
                        model_series = match.group(1)
                    else:
                        # 如果文件名不包含模型系列名，尝试从目录名获取
                        dir_match = re.search(r'(LexJade6-[A-Za-z0-9\-]+)', model_series_dir)
                        if dir_match:
                            model_series = dir_match.group(1)
                    if model_series and model_series in MODEL_CONFIG_MAP:
                        available_models.append({
                            'series': model_series,
                            'config_class': MODEL_CONFIG_MAP[model_series],
                            'path': full_path,
                            'mode': mode,
                            'filename': file
                        })
                    else:
                        print(f"警告: 无法从文件名 '{file}' 或目录名 '{model_series_dir}' 推断出有效的模型系列。")
    return available_models

def load_model_from_ckpt(ckpt_path, config_class, device, mode="Pretrain"):
    """从 .pth 检查点文件加载模型"""
    print(f"正在加载模型: {ckpt_path}")
    print(f"模型系列: {config_class.__name__}, 推断模式: {mode}")
    try:
        config = config_class()
        model = LexJadeForCausalLM(config)
        # 加载检查点
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint
        # 如果检查点是字典格式，尝试获取 'state_dict' 或 'model' 键
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            # 可以在这里添加更多键的检查
        # 加载状态字典
        # strict=False 允许部分匹配，这对于微调或结构略有不同的模型很有用
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"警告: 加载时发现缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 加载时发现意外的键: {unexpected_keys}")
        print("模型权重加载成功。")
        model.to(device)
        model.eval()
        # --- 修改点 1: 改进 Tokenizer 查找逻辑 ---
        tokenizer = None
        # 1. 从模型路径推断模型系列名称
        model_series_name = None
        for series_name in MODEL_CONFIG_MAP.keys():
            # 使用更宽松的匹配，检查模型系列名是否是 ckpt_path 的子串
            if series_name in ckpt_path:
                model_series_name = series_name
                break
        # 2. 构建可能的 tokenizer 路径列表
        # 优先查找 /root/minimind/model 下的特定 tokenizer 和其变体
        tokenizer_paths_to_check = []
        if model_series_name:
            # 添加精确匹配的路径 (例如 tokenizer_LexJade6-Flash)
            tokenizer_paths_to_check.append(os.path.join("/root/minimind/model", f"tokenizer_{model_series_name}"))
            # 添加可能的变体路径 (例如 tokenizer_LexJadeLM6-Flash)
            # 这是为了兼容您实际的目录名 tokenizer_LexJadeLM6-Flash
            variant_name = model_series_name.replace("6", "LM6")
            if variant_name != model_series_name:
                tokenizer_paths_to_check.append(os.path.join("/root/minimind/model", f"tokenizer_{variant_name}"))
        # 添加其他备选路径
        tokenizer_paths_to_check.extend([
            os.path.join("/root/minimind/model", "tokenizer_LexJadeLM6"),          # 通用 tokenizer
            os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "tokenizer"), # 同级 tokenizer 目录
            os.path.join(os.path.dirname(ckpt_path), "tokenizer"),                  # 同级 tokenizer 目录
            os.path.join(project_root, "tokenizer"),                                # 项目根目录下的 tokenizer
            os.path.join(project_root, os.path.basename(os.path.dirname(ckpt_path)), "tokenizer"), # 模型系列名下的 tokenizer
        ])
        # 3. 遍历候选路径尝试加载
        for path in tokenizer_paths_to_check:
            if os.path.exists(path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    print(f"Tokenizer 从 '{path}' 加载成功。")
                    break
                except Exception as e:
                    print(f"尝试从 '{path}' 加载 tokenizer 失败: {e}")
        # 4. 如果所有路径都失败，则报错
        if tokenizer is None:
            raise FileNotFoundError(
                f"无法在以下路径找到合适的 tokenizer: {tokenizer_paths_to_check}。"
                f"请确保对应模型系列的 tokenizer 存在于 '/root/minimind/model/' 目录下，"
                f"例如 '/root/minimind/model/tokenizer_LexJadeLM6-Flash' 或 '/root/minimind/model/tokenizer_LexJade6-Flash'。"
            )
        # --- 修改点 1 结束 ---
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params / 1e6:.3f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.3f}M")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def init_model(args):
    """初始化模型和分词器"""
    # --- 关键修改: 确保 device 在 init_model 中也设置 ---
    if not hasattr(args, 'device') or args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = device
    # --- 关键修改结束 ---
    model = None
    tokenizer = None
    if args.load_mode == 0:
        # 自动扫描模式已在 main 函数中处理
        # 这里应该不会被调用到，但为了健壮性保留
        print("错误：init_model 不应在 load_mode=0 时被直接调用。")
        return None, None
    elif args.load_mode == 1:
        # 加载 Hugging Face 格式模型 (目前未实现，可扩展)
        print("load_mode=1 (Hugging Face 格式) 暂未实现。")
        return None, None
    elif args.load_mode == 2:
        # 手动指定 .pth 路径
        ckpt_path = args.manual_ckpt_path
        if not os.path.exists(ckpt_path):
            print(f"指定的检查点文件不存在: {ckpt_path}")
            return None, None
        # 从路径推断模型系列
        model_series = None
        for series_name in MODEL_CONFIG_MAP.keys():
            if series_name in ckpt_path:
                model_series = series_name
                break
        if not model_series:
            print(f"无法从路径 '{ckpt_path}' 推断出模型系列。请确保路径包含有效的模型系列名称。")
            return None, None
        config_class = MODEL_CONFIG_MAP[model_series]
        # 推断模式
        mode = "Pretrain" # 默认
        if 'sft' in ckpt_path.lower() or 'chat' in ckpt_path.lower() or 'sft_out' in ckpt_path:
            mode = "SFT/Chat"
        model, tokenizer = load_model_from_ckpt(ckpt_path, config_class, args.device, mode)
    else:
        print(f"无效的 --load_mode 值: {args.load_mode}")
    return model, tokenizer

def get_prompt_datas(args):
    """获取测试提示数据"""
    # 可以从文件或直接定义
    default_prompts = [
        "马克思主义基本原理",
        "人类大脑的主要功能",
        "万有引力原理是",
        "世界上最高的山峰是",
        "二氧化碳在空气中",
        "地球上最大的动物有",
        "杭州市的美食有"
    ]
    return default_prompts

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with LexJadeLM6")
    # - 新增/修改的参数 -
    parser.add_argument('--load_mode', default=0, type=int,
                        help="0: 自动扫描并选择模型 (默认), 1: 加载 Hugging Face 格式模型, 2: 手动指定 .pth 路径")
    parser.add_argument('--manual_ckpt_path', default='', type=str,
                        help="当 load_mode=2 时，指定 .pth 检查点文件的完整路径")
    parser.add_argument('--history_cnt', default=0, type=int,
                        help="保留的历史对话轮数 (默认为0，即不保留历史)")
    parser.add_argument('--max_seq_len', default=512, type=int,
                        help="最大序列长度 (默认512)")
    # - 添加生成参数 -
    parser.add_argument('--temperature', type=float, default=0.5,
                        help="控制随机性，值越大输出越随机 (0.1-2.0)")
    parser.add_argument('--top_p', type=float, default=0.9,
                        help="核采样参数，累积概率阈值 (0.0-1.0)")
    parser.add_argument('--top_k', type=int, default=45,
                        help="限制考虑的最高概率词汇数量")
    parser.add_argument('--repetition_penalty', type=float, default=1.25,
                        help="重复惩罚系数，大于1.0 (1.0-2.0)")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3,
                        help="防止n-gram重复的大小")
    parser.add_argument('--min_new_tokens', type=int, default=1,
                        help="最小生成token数")
    parser.add_argument('--early_stopping', action='store_true',
                        help="当生成EOS时提前停止")
    args = parser.parse_args()
    # - 模型加载 -
    # --- 修改: 先选择模型类型再扫描 ---
    base_out_dir = get_base_out_dir()
    # --- 修改结束 ---
    model = None
    tokenizer = None
    # --- 关键修改: 在所有分支之前设置 device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # --- 关键修改结束 ---
    if args.load_mode == 0:
        # 自动扫描并选择模型
        available_models = find_available_models(base_out_dir)
        if not available_models:
            print("在指定目录下未找到任何可用的模型检查点。")
            return
        print("\n--- 可用的模型检查点 ---")
        for i, model_info in enumerate(available_models):
            print(f"[{i}] {os.path.basename(model_info['path'])} (模式: {model_info['mode']})")
        try:
            choice = int(input("\n请选择要加载的模型编号: "))
            if 0 <= choice < len(available_models):
                selected_model_info = available_models[choice]
                # 保存模式信息，供后续使用
                args.model_mode_from_scan = selected_model_info['mode']
                model, tokenizer = load_model_from_ckpt(
                    selected_model_info['path'],
                    selected_model_info['config_class'],
                    args.device, # 使用已设置的 device
                    selected_model_info['mode']
                )
            else:
                print("无效的选择，程序退出。")
                return
        except ValueError:
            print("请输入一个有效的数字，程序退出。")
            return
    elif args.load_mode in [1, 2]:
        model, tokenizer = init_model(args)
    else:
        print(f"无效的 --load_mode 值: {args.load_mode}")
        return
    if model is None or tokenizer is None:
        print("模型或分词器加载失败，程序退出。")
        return
    # - 测试流程 -
    prompts = get_prompt_datas(args)
    while True:
        try:
            test_mode = int(input('\n--- 请选择测试模式 ---\n[0] 自动测试\n[1] 手动输入\n请输入选项 (0 或 1): '))
            if test_mode in [0, 1]:
                break
            else:
                print("无效输入，请输入 0 或 1。")
        except ValueError:
            print("请输入一个有效的数字。")
    # 初始化对话历史
    messages = []
    # 判断是否为 Pretrain 模式
    # --- 修改: 根据用户选择的模型类型或扫描结果确定模式 ---
    is_pretrain_mode = getattr(args, 'model_mode_from_scan', 'Pretrain') == 'Pretrain'
    if args.load_mode == 2: # 如果是手动加载，再次确认模式
         is_pretrain_mode = 'sft' not in args.manual_ckpt_path.lower() and 'chat' not in args.manual_ckpt_path.lower() and 'sft_out' not in args.manual_ckpt_path
    # --- 修改结束 ---
    print("\n--- 开始测试 ---")
    if test_mode == 0:
        # 自动测试模式
        for prompt in prompts:
            print(f"\n👶: {prompt}")
            # - 修复点 1: 历史消息处理 -
            if is_pretrain_mode:
                messages = [] # 清空历史，确保独立生成
            else:
                if args.history_cnt > 0:
                    messages = messages[-args.history_cnt:]
                else:
                    messages = []
            # - 修复点 1 结束 -
            messages.append({"role": "user", "content": prompt})
            # - 修复点 2: 简化模式判断和输入构建 -
            if not is_pretrain_mode:
                try:
                    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    print(f"应用聊天模板失败 ({e})，将使用简单拼接。")
                    new_prompt = ""
                    for msg in messages:
                        if msg["role"] == "user":
                            new_prompt += f"<|user|>{msg['content']}"
                        elif msg["role"] == "assistant":
                            new_prompt += f"<|assistant|>{msg['content']}"
                    new_prompt += "<|assistant|>"
            else:
                # Pretrain 模式使用最简单的形式
                new_prompt = prompt # tokenizer.bos_token + prompt
            # - 修复点 2 结束 -
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
            if inputs["input_ids"].size(1) == args.max_seq_len:
                print("警告: 输入已达到最大长度限制。")
            print('🤖️: ', end='')

            # --- 启用流式输出 ---
            # 创建 TextStreamer 实例
            # 注意：skip_prompt=True 避免打印输入提示词, skip_special_tokens=True 移除特殊token
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            # --- 启用流式输出结束 ---

            # 模型生成
            try:
                with torch.no_grad():
                    # --- 修改 generate 调用以启用流式输出 ---
                    # 移除或调整 max_new_tokens 的计算方式，确保逻辑兼容
                    generated_ids = model.generate(
                        inputs["input_ids"],
                        # max_new_tokens=min(512, args.max_seq_len - inputs["input_ids"].size(1)), # 可以保留
                        max_new_tokens=args.max_seq_len - inputs["input_ids"].size(1) if inputs["input_ids"].size(1) < args.max_seq_len else 512, # 确保 max_new_tokens > 0
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_new_tokens=args.min_new_tokens,
                        early_stopping=args.early_stopping,
                        attention_mask=inputs["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer, # <-- 启用流式输出
                    )
                    # --- 流式输出下，后续处理需要调整 ---
                    # 因为 TextStreamer 已经打印了输出，通常不需要再 decode 和 print response
                    # 但如果你需要将 response 存入 messages 或做其他处理，仍需 decode

                    # --- 保留调试信息 (可选) ---
                    # print(f"DEBUG: inputs['input_ids'].shape = {inputs['input_ids'].shape}")
                    # print(f"DEBUG: inputs['input_ids'].dtype = {inputs['input_ids'].dtype}")
                    # print(f"DEBUG: generated_ids.shape = {generated_ids.shape}")
                    # print(f"DEBUG: generated_ids.dtype = {generated_ids.dtype}")

                    # --- 保留历史记录和 decode (如果需要) ---
                    # 检查维度和长度 (这部分逻辑可能需要根据 streamer 的行为微调，但基础检查保留)
                    if generated_ids.dim() != 2:
                         # 可以改为警告，因为 streamer 可能改变了行为，或者确保 generate 正常返回
                         print(f"Warning: Expected generated_ids to be 2-dimensional, but got {generated_ids.dim()}-dimensional tensor with shape {generated_ids.shape}. Streamer might be in use.")
                         # 为了后续处理，假设维度是正确的或尝试恢复
                         if generated_ids.dim() == 1:
                             generated_ids = generated_ids.unsqueeze(0) # 尝试调整为 [1, seq_len]
                         else:
                             response = "<生成错误: 张量维度异常>"
                             if not is_pretrain_mode:
                                 messages.append({"role": "assistant", "content": response})
                             # print(response) # 如果 decode 失败，打印错误 (非流式)
                             print('\n' + '-' * 20)
                             continue # 跳过本轮循环的剩余部分

                    input_len = inputs["input_ids"].shape[1]
                    total_len = generated_ids.shape[1]

                    if input_len > total_len:
                        print(f"Warning: Input length ({input_len}) seems greater than generated total length ({total_len}). Streamer might be in use or generation failed.")
                        response = "<生成错误: 长度异常>"
                        if not is_pretrain_mode:
                            messages.append({"role": "assistant", "content": response})
                        # print(response) # 打印错误信息 (非流式)
                        print('\n' + '-' * 20)
                        continue
                    elif input_len == total_len:
                        # print("Warning: Input length equals total generated length. No new tokens generated?") # 警告可选
                        response = "" # 空响应
                    else:
                        # 安全地切片并解码
                        generated_token_ids = generated_ids[0, input_len:] # Shape: [new_sequence_length]
                        # 注意：即使使用了 streamer，我们仍然可以 decode 来获取完整的 response 字符串用于历史记录
                        response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                    # --- 将响应添加到历史记录 (如果适用) ---
                    if not is_pretrain_mode:
                        messages.append({"role": "assistant", "content": response})
                    # --- 流式输出下，通常不在此处打印 response ---
                    # print(response, end='') # <-- 注释掉或删除这行

            except Exception as e:
                print(f"生成过程中出错: {e}")
                import traceback
                traceback.print_exc()
                response = "<生成错误>"
                # 即使出错，也打印错误信息 (非流式)
                # print(response, end='')

            # --- 在流式输出后，打印分隔符 ---
            print('\n' + '-' * 20) # 在每次响应后打印分隔符

    else:
        # 手动输入模式
        while True:
            try:
                user_input = input("\n👶: ")
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("再见！")
                    break
                prompt = user_input
                # - 修复点 1: 历史消息处理 (同上) -
                if is_pretrain_mode:
                    messages = []
                else:
                    if args.history_cnt > 0:
                        messages = messages[-args.history_cnt:]
                    else:
                        messages = []
                # - 修复点 1 结束 -
                messages.append({"role": "user", "content": prompt})
                # - 修复点 2: 简化模式判断和输入构建 (同上) -
                if not is_pretrain_mode:
                    try:
                        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception as e:
                        print(f"应用聊天模板失败 ({e})，将使用简单拼接。")
                        new_prompt = ""
                        for msg in messages:
                            if msg["role"] == "user":
                                new_prompt += f"<|user|>{msg['content']}"
                            elif msg["role"] == "assistant":
                                new_prompt += f"<|assistant|>{msg['content']}"
                        new_prompt += "<|assistant|>"
                else:
                    new_prompt = prompt
                # - 修复点 2 结束 -
                inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
                if inputs["input_ids"].size(1) == args.max_seq_len:
                    print("警告: 输入已达到最大长度限制。")
                print('🤖️: ', end='')

                 # --- 启用流式输出 ---
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                # --- 启用流式输出结束 ---

                # 模型生成
                try:
                    with torch.no_grad():
                        # --- 修改 generate 调用以启用流式输出 ---
                        generated_ids = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=args.max_seq_len - inputs["input_ids"].size(1) if inputs["input_ids"].size(1) < args.max_seq_len else 512,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            min_new_tokens=args.min_new_tokens,
                            early_stopping=args.early_stopping,
                            attention_mask=inputs["attention_mask"],
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            streamer=streamer, # <-- 启用流式输出
                        )

                        # --- 保留历史记录和 decode (如果需要) ---
                        if generated_ids.dim() != 2:
                             print(f"Warning: Expected generated_ids to be 2-dimensional, but got {generated_ids.dim()}-dimensional tensor with shape {generated_ids.shape}. Streamer might be in use.")
                             if generated_ids.dim() == 1:
                                 generated_ids = generated_ids.unsqueeze(0)
                             else:
                                 response = "<生成错误: 张量维度异常>"
                                 if not is_pretrain_mode:
                                     messages.append({"role": "assistant", "content": response})
                                 # print(response) # 打印错误信息 (非流式)
                                 print('\n' + '-' * 20)
                                 continue

                        input_len = inputs["input_ids"].shape[1]
                        total_len = generated_ids.shape[1]

                        if input_len > total_len:
                            print(f"Warning: Input length ({input_len}) seems greater than generated total length ({total_len}). Streamer might be in use or generation failed.")
                            response = "<生成错误: 长度异常>"
                            if not is_pretrain_mode:
                                messages.append({"role": "assistant", "content": response})
                            # print(response) # 打印错误信息 (非流式)
                            print('\n' + '-' * 20)
                            continue
                        elif input_len == total_len:
                            # print("Warning: Input length equals total generated length. No new tokens generated?") # 警告可选
                            response = ""
                        else:
                            generated_token_ids = generated_ids[0, input_len:]
                            response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                        if not is_pretrain_mode:
                            messages.append({"role": "assistant", "content": response})
                        # print(response, end='') # <-- 注释掉或删除这行

                except Exception as e:
                    print(f"生成过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
                    response = "<生成错误>"
                    # print(response, end='') # 打印错误信息 (非流式)

                print('\n' + '-' * 20) # 在每次响应后打印分隔符
            except KeyboardInterrupt:
                print("\n收到中断信号，退出程序。")
                break
            except Exception as e:
                print(f"\n发生未预期的错误: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()