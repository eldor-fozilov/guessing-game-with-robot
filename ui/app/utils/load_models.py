# load config
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from ultralytics import YOLO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def load_yolo_model(model_name="models/yolo11n.pt", device='cpu'):
    yolo_model = YOLO(model_name)
    yolo_model.fuse()
    yolo_model.to(device)
    return yolo_model


def load_llm_model(model_name="meta-llama/Llama-3.2-1B-Instruct", device="cpu"):

    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype="auto",
                                                     device_map=device)

    llm_model.eval()

    return llm_model, llm_tokenizer


def load_vlm_model(model_name="OpenGVLab/InternVL2_5-2B", load_in_8bit=False, device="cpu"):

    vlm_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True)
    vlm_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False)

    vlm_model.to(device).eval()

    return vlm_model, vlm_tokenizer


def load_yolo_world_model(model_path="./YOLO-World/pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth",
                          config_path="./YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py",
                          device="cpu"):

    cfg = Config.fromfile(
        config_path
    )
    cfg.work_dir = "./YOLO-World/"
    cfg.load_from = model_path
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)

    # run model evaluation
    runner.model.to(device).eval()

    return runner
