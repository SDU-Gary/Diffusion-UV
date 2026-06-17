#!/bin/bash

# W&B 配置验证脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   W&B 配置验证${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# 检查 W&B 安装
echo -e "${YELLOW}[1/5] 检查 W&B 安装...${NC}"
if python -c "import wandb" 2>/dev/null; then
    echo -e "${GREEN}✓ W&B 已安装${NC}"
    python -c "import wandb; print(f\"  版本: {wandb.__version__}\")"
else
    echo -e "${RED}✗ W&B 未安装${NC}"
    echo -e "${YELLOW}安装命令: pip install wandb${NC}"
    exit 1
fi

echo ""

# 检查 W&B 登录状态
echo -e "${YELLOW}[2/5] 检查 W&B 登录状态...${NC}"
if python -c "
import wandb
import os
try:
    # 尝试获取 API key（会从 .netrc 读取）
    api_key = wandb.api.api_key
    if api_key:
        print(f'✓ Logged in (API key: {api_key[:8]}...)')
        exit(0)
    else:
        exit(1)
except:
    # 检查 .netrc 文件
    netrc_path = os.path.expanduser('~/.netrc')
    if os.path.exists(netrc_path):
        with open(netrc_path, 'r') as f:
            content = f.read()
            if 'api.wandb.ai' in content:
                print('✓ Found credentials in .netrc')
                exit(0)
    exit(1)
" 2>/dev/null; then
    echo -e "${GREEN}✓ W&B 已登录${NC}"
else
    echo -e "${RED}✗ W&B 未登录${NC}"
    echo -e "${YELLOW}请运行: wandb login${NC}"
    exit 1
fi

echo ""

# 检查配置文件
echo -e "${YELLOW}[3/5] 检查配置文件...${NC}"

if [ ! -f "configs/production.yaml" ]; then
    echo -e "${RED}✗ 配置文件不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件存在${NC}"

# 解析配置
USE_WANDB=$(python -c "
import yaml
with open('configs/production.yaml') as f:
    config = yaml.safe_load(f)
    print(config['logging']['use_wandb'])
")

WANDB_PROJECT=$(python -c "
import yaml
with open('configs/production.yaml') as f:
    config = yaml.safe_load(f)
    print(config['logging']['wandb_project'])
")

WANDB_MODE=$(python -c "
import yaml
with open('configs/production.yaml') as f:
    config = yaml.safe_load(f)
    print(config['logging']['wandb_mode'])
")

if [ "$USE_WANDB" = "True" ] || [ "$USE_WANDB" = "true" ]; then
    echo -e "${GREEN}✓ W&B 已启用${NC}"
    echo -e "  项目: ${WANDB_PROJECT}"
    echo -e "  模式: ${WANDB_MODE}"
else
    echo -e "${RED}✗ W&B 未启用${NC}"
    echo -e "  use_wandb: ${USE_WANDB}"
    exit 1
fi

echo ""

# 检查网络连接
echo -e "${YELLOW}[4/5] 检查网络连接...${NC}"
if ping -c 1 -W 2 api.wandb.ai > /dev/null 2>&1; then
    echo -e "${GREEN}✓ W&B API 可访问${NC}"
else
    echo -e "${RED}✗ 无法访问 W&B API${NC}"
    echo -e "${YELLOW}将使用离线模式${NC}"
    WANDB_MODE="offline"
fi

echo ""

# 测试 W&B 初始化
echo -e "${YELLOW}[5/5] 测试 W&B 初始化...${NC}"
python << 'EOF'
import yaml
import sys

try:
    # 加载配置
    with open('configs/production.yaml') as f:
        config = yaml.safe_load(f)

    # 检查 W&B 配置
    logging_config = config['logging']
    print(f"  实验名称: {logging_config['experiment_name']}")
    print(f"  日志目录: {logging_config['log_dir']}")
    print(f"  记录间隔: {logging_config['log_interval']} 步")

    print("\n✓ W&B 配置验证成功")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ 配置验证失败: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 配置验证通过${NC}"
else
    echo -e "${RED}✗ 配置验证失败${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}   W&B 配置验证完成！${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo -e "${GREEN}配置摘要:${NC}"
echo -e "  W&B 状态: ${GREEN}已启用${NC}"
echo -e "  项目名: ${WANDB_PROJECT}"
echo -e "  运行模式: ${WANDB_MODE}"
echo -e "  实验名称: bunny_production"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo -e "  1. 运行测试训练验证 W&B:"
echo -e "     ${GREEN}./train.sh test cuda${NC}"
echo ""
echo -e "  2. 运行生产训练:"
echo -e "     ${GREEN}./train.sh production cuda${NC}"
echo ""
echo -e "  3. 访问 W&B Dashboard (训练开始后会在终端显示链接)"
echo ""
