#!/bin/bash

# Diffusion-UV Training Scripts
# 快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Diffusion-UV Training Launcher${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: ./train.sh <mode> [device]"
    echo ""
    echo "可选模式:"
    echo "  test        - 测试模式 (5+2+2 epochs)"
    echo "  production  - 生产模式 (500+200+100 epochs)"
    echo ""
    echo "可选设备:"
    echo "  cuda        - 使用 GPU (默认)"
    echo "  cpu         - 使用 CPU"
    echo ""
    echo "示例:"
    echo "  ./train.sh test cuda"
    echo "  ./train.sh production cpu"
    exit 1
fi

MODE=$1
DEVICE=${2:-cuda}

# 根据模式选择配置
if [ "$MODE" = "test" ]; then
    CONFIG="configs/gpu_training.yaml"
    echo -e "${YELLOW}模式: 测试训练 (9 epochs)${NC}"
elif [ "$MODE" = "production" ]; then
    if [ "$DEVICE" = "cpu" ]; then
        CONFIG="configs/production_cpu.yaml"
    else
        CONFIG="configs/production.yaml"
    fi
    echo -e "${YELLOW}模式: 生产训练 (800 epochs)${NC}"
else
    echo -e "${RED}错误: 未知模式 '$MODE'${NC}"
    echo "可选模式: test, production"
    exit 1
fi

echo -e "${YELLOW}配置: $CONFIG${NC}"
echo -e "${YELLOW}设备: $DEVICE${NC}"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

# 检查数据文件
if [ ! -f "data/models/stanford-bunny.obj" ]; then
    echo -e "${RED}错误: 模型文件不存在: data/models/stanford-bunny.obj${NC}"
    exit 1
fi

# 显示训练信息
echo -e "${GREEN}训练信息:${NC}"
python -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
    print(f\"  Phase 1: {config['training']['phase1_epochs']} epochs\")
    print(f\"  Phase 2: {config['training']['phase2_epochs']} epochs\")
    print(f\"  Phase 3: {config['training']['phase3_epochs']} epochs\")
    print(f\"  Total: {config['training']['phase1_epochs'] + config['training']['phase2_epochs'] + config['training']['phase3_epochs']} epochs\")
    print(f\"  Batch size (Phase 1): {config['training']['batch_size_phase1']}\")
    print(f\"  Samples per epoch: {config['data']['num_samples_per_epoch']:,}\")
"
echo ""

# 询问确认
if [ "$MODE" = "production" ]; then
    echo -e "${YELLOW}生产训练将需要较长时间:${NC}"
    if [ "$DEVICE" = "cuda" ]; then
        echo -e "  预计时间: 10-15 分钟"
    else
        echo -e "  预计时间: 1-1.5 小时"
    fi
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}训练已取消${NC}"
        exit 0
    fi
fi

# 清理旧的缓存（可选）
read -p "是否清理旧缓存? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CACHE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['cache_dir'])")
    if [ -d "$CACHE_DIR" ]; then
        echo -e "${YELLOW}清理缓存: $CACHE_DIR${NC}"
        rm -rf "$CACHE_DIR"
    fi
fi

echo ""
echo -e "${GREEN}开始训练...${NC}"
echo ""

# 运行训练
python scripts/train.py --config "$CONFIG" --device "$DEVICE"

# 训练完成
echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   训练完成!${NC}"
echo -e "${GREEN}==================================================${NC}"

# 显示输出文件
LOG_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['log_dir'])")
EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['experiment_name'])")

echo ""
echo -e "${GREEN}生成的文件:${NC}"
echo "  检查点: $LOG_DIR/$EXP_NAME/checkpoints/"
echo "  缓存: $(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['cache_dir'])")"
echo ""

ls -lh "$LOG_DIR/$EXP_NAME/checkpoints/" 2>/dev/null || echo "  (没有检查点文件)"

echo ""
echo -e "${GREEN}训练日志已保存到: $LOG_DIR${NC}"
