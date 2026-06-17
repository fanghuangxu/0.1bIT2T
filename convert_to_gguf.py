"""
NextAI → GGUF 转换器
将 NextAI 模型转换为 GGUF 格式
支持 LSTM 和 Transformer 架构
"""
import os
import sys
import struct
import numpy as np
import torch


# GGUF 格式常量
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# 张量数据类型
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


class GGUFWriter:
    """GGUF 文件写入器 (v3)"""

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "wb")
        self.metadata = []
        self.tensor_info = []
        self.alignment = 32

    def add_string(self, key: str, value: str):
        self.metadata.append(("string", key, value.encode("utf-8")))

    def add_uint32(self, key: str, value: int):
        self.metadata.append(("uint32", key, value))

    def add_float32(self, key: str, value: float):
        self.metadata.append(("float32", key, value))

    def add_array_f32(self, key: str, values):
        self.metadata.append(("array_f32", key, values))

    def add_array_i32(self, key: str, values):
        self.metadata.append(("array_i32", key, values))

    def add_tensor(self, name: str, tensor: torch.Tensor):
        data = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        self.tensor_info.append({
            "name": name,
            "shape": list(data.shape),
            "type": GGML_TYPE_F32,
            "data": data.tobytes(),
        })

    def _type_id(self, mtype: str) -> int:
        return {
            "string": 8,
            "uint32": 4,
            "float32": 6,
            "array_f32": 9,
            "array_i32": 10,
        }[mtype]

    def _write_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def write(self):
        """写入 GGUF 文件"""
        self.file.write(GGUF_MAGIC)
        self.file.write(struct.pack("<I", GGUF_VERSION))
        self.file.write(struct.pack("<Q", len(self.tensor_info)))
        self.file.write(struct.pack("<Q", len(self.metadata)))

        for mtype, key, value in self.metadata:
            self.file.write(self._write_string(key))
            self.file.write(struct.pack("<I", self._type_id(mtype)))
            if mtype == "string":
                self.file.write(struct.pack("<Q", len(value)))
                self.file.write(value)
            elif mtype == "uint32":
                self.file.write(struct.pack("<I", value))
            elif mtype == "float32":
                self.file.write(struct.pack("<f", value))
            elif mtype == "array_f32":
                self.file.write(struct.pack("<I", len(value)))
                for v in value:
                    self.file.write(struct.pack("<f", v))
            elif mtype == "array_i32":
                self.file.write(struct.pack("<I", len(value)))
                for v in value:
                    self.file.write(struct.pack("<i", v))

        for t in self.tensor_info:
            self.file.write(self._write_string(t["name"]))
            self.file.write(struct.pack("<I", len(t["shape"])))
            for dim in t["shape"]:
                self.file.write(struct.pack("<Q", dim))
            self.file.write(struct.pack("<I", t["type"]))
            self.file.write(struct.pack("<Q", 0))

        current = self.file.tell()
        pad = (self.alignment - (current % self.alignment)) % self.alignment
        self.file.write(b"\x00" * pad)

        for t in self.tensor_info:
            self.file.write(t["data"])

        self.file.close()
        print(f"GGUF 文件已写入: {self.path}")


def convert_to_gguf(pt_path: str, gguf_path: str):
    """转换主函数"""
    print("=" * 60)
    print("NextAI → GGUF 转换")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1] 加载模型: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    # 显示checkpoint结构
    print(f"  顶层键: {list(checkpoint.keys())}")

    # 提取配置
    cfg = checkpoint.get("cfg", {})
    model_type = checkpoint.get("model_type", "unknown")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    tokenizer = checkpoint.get("tokenizer", {})

    print(f"  模型类型: {model_type}")
    print(f"  配置: {cfg}")
    print(f"  张量数量: {len(state_dict)}")

    # 2. 推断模型配置
    print("\n[2] 推断模型配置")

    vocab_size = cfg.get("vocab_size", 260)
    d_model = cfg.get("d_model", 320)
    hidden_size = cfg.get("hidden_size", 384)
    num_layers = cfg.get("num_layers", 1)
    dropout = cfg.get("dropout", 0.1)
    max_len = cfg.get("max_len", 128)

    # 尝试从权重中推断
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "embed" in key.lower() and "weight" in key:
            if len(tensor.shape) == 2:
                vocab_size, d_model = tensor.shape
                break

    print(f"  vocab_size:   {vocab_size}")
    print(f"  d_model:      {d_model}")
    print(f"  hidden_size:  {hidden_size}")
    print(f"  num_layers:   {num_layers}")
    print(f"  max_len:      {max_len}")

    # 3. 创建 GGUF 文件
    print(f"\n[3] 创建 GGUF: {gguf_path}")
    writer = GGUFWriter(gguf_path)

    # 3.1 基本元数据
    writer.add_string("general.name", "NextAI")
    writer.add_string("general.author", "Next Studio")
    writer.add_string("general.version", "0.1.0")
    writer.add_string("general.description", f"NextAI 0.1B multilingual chat model ({model_type}) - EN/DE/ZH")
    writer.add_string("general.license", "Apache 2.0")
    writer.add_string("general.file_type", "f32")

    # 3.2 模型架构
    writer.add_string("model.type", "nextai")
    writer.add_string("model.architecture", f"nextai-{model_type}")
    writer.add_uint32("model.block_count", num_layers)
    writer.add_uint32("model.embedding_length", d_model)
    writer.add_uint32("model.feed_forward_length", hidden_size)
    writer.add_uint32("model.context_length", max_len)
    writer.add_uint32("nextai.vocab_size", vocab_size)

    # 3.3 Tokenizer
    writer.add_string("tokenizer.ggml.model", "byte")
    writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.get("bos", 1))
    writer.add_uint32("tokenizer.ggml.eos_token_id", tokenizer.get("eos", 2))
    writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.get("pad", 0))
    writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.get("unk", 3))
    writer.add_uint32("tokenizer.ggml.vocab_size", vocab_size)

    # 4. 写入张量
    print(f"\n[4] 转换张量...")

    tensor_count = 0
    total_params = 0
    skipped = 0

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            skipped += 1
            continue

        # 跳过非权重
        if any(skip in key for skip in ["num_batches", "step", "_metadata"]):
            skipped += 1
            continue

        writer.add_tensor(key, tensor)
        tensor_count += 1
        total_params += tensor.numel()

        if tensor_count <= 8 or tensor_count % 10 == 0:
            print(f"  + {key}: {list(tensor.shape)} ({tensor.numel()} params)")

    print(f"\n  总张量: {tensor_count} (跳过 {skipped})")
    print(f"  总参数: {total_params / 1e6:.3f}M ({total_params})")

    # 5. 写入文件
    print(f"\n[5] 写入文件...")
    writer.write()

    # 6. 报告
    file_size = os.path.getsize(gguf_path)
    print(f"\n" + "=" * 60)
    print(f"转换完成!")
    print(f"=" * 60)
    print(f"  输出文件: {gguf_path}")
    print(f"  文件大小: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")
    print(f"  张量数量: {tensor_count}")
    print(f"  参数规模: {total_params / 1e6:.3f}M")
    print(f"  模型类型: {model_type}")

    return gguf_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NextAI → GGUF 转换器")
    parser.add_argument("--input", "-i", required=True, help="输入 .pt 文件")
    parser.add_argument("--output", "-o", required=True, help="输出 .gguf 文件")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到 {args.input}")
        sys.exit(1)

    convert_to_gguf(args.input, args.output)


if __name__ == "__main__":
    main()
