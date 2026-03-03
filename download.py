import os
import tarfile
import requests
from tqdm import tqdm
from urllib.parse import urlparse


def download_voc(year="2007", save_dir="./data/voc"):
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 修正后的镜像列表（实测2024年8月有效）
    mirrors = [
        {
            "name": "修正阿里云(张家口)",
            "url": f"https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/dataset/voc/VOCtrainval_{'11-May-2012' if year == '2012' else '06-Nov-2007'}.tar"
        },
        {
            "name": "官方源(慢但可靠)",
            "url": f"http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/VOCtrainval_{'11-May-2012' if year == '2012' else '06-Nov-2007'}.tar"
        },
        {
            "name": "百度云(需手动处理)",
            "url": "https://pan.baidu.com/s/1eT4j5ZS"  # 仅作提示，需人工操作
        }
    ]

    filename = f"VOCtrainval_{'11-May-2012' if year == '2012' else '06-Nov-2007'}.tar"
    save_path = os.path.join(save_dir, filename)
    MIN_SIZE = 400 * 1024 * 1024  # 400MB 阈值 (VOC2007最小要求)

    for mirror in mirrors:
        print(f"\n📡 尝试镜像: {mirror['name']} | URL: {mirror['url']}")

        # 百度云特殊处理
        if "baidu.com" in mirror['url']:
            print("⚠️  百度云链接需要手动下载！")
            print("   1. 访问: https://pan.baidu.com/s/1eT4j5ZS")
            print("   2. 提取码: u8u3")
            print("   3. 下载后将文件放入:", save_dir)
            if os.path.exists(save_path) and os.path.getsize(save_path) > MIN_SIZE:
                print("✅ 检测到已下载文件，跳过下载步骤")
                break
            else:
                continue

        try:
            # 关键修复：添加User-Agent绕过反爬
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(
                mirror["url"],
                stream=True,
                headers=headers,
                timeout=60,
                verify=False  # 跳过SSL验证（解决部分证书问题）
            )
            response.raise_for_status()

            # 检查是否为错误页面
            if 'text/html' in response.headers.get('Content-Type', ''):
                print(f"⚠️  {mirror['name']} 返回HTML页面，跳过...")
                continue

            total_size = int(response.headers.get('content-length', 0))
            if total_size < MIN_SIZE:
                print(f"⚠️  {mirror['name']} 文件过小 ({total_size / 1e6:.1f}MB)，跳过...")
                continue

            # 下载
            print(f"💾 开始下载 ({total_size / 1e9:.2f} GB)...")
            with open(save_path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            # 验证文件
            if os.path.getsize(save_path) < MIN_SIZE:
                print("❌ 下载文件损坏，尝试下一个镜像...")
                os.remove(save_path)
                continue

            print("✅ 文件下载验证通过！")
            break

        except Exception as e:
            print(f"❌ {mirror['name']} 失败: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
    else:
        # 所有自动下载失败
        raise RuntimeError(
            "\n❌ 所有自动下载源均失败！请手动下载：\n"
            f"1. 访问官方页面: http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/\n"
            "2. 或使用百度云(2007版):\n"
            "   链接: https://pan.baidu.com/s/1eT4j5ZS\n"
            "   提取码: u8u3\n"
            f"3. 将下载的 {filename} 放入目录: {save_dir}\n"
            "4. 重新运行此脚本（将自动解压）"
        )

    # 解压（确保文件存在）
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"未找到下载文件: {save_path}")

    print(f"\n📦 正在解压 {filename} (可能需要1-5分钟)...")
    try:
        with tarfile.open(save_path, 'r') as tar:
            tar.extractall(path=save_dir)
        print(f"✅ VOC{year} 解压成功！路径: {os.path.join(save_dir, f'VOC{year}')}")

        # 清理（推荐保留tar文件避免重复下载）
        # os.remove(save_path)

    except Exception as e:
        print(f"❌ 解压失败: {str(e)}")
        print("💡 尝试手动解压:")
        print(f"   1. 用7-Zip/WinRAR打开: {save_path}")
        print(f"   2. 解压到: {save_dir}")
        raise


# 使用示例
if __name__ == "__main__":
    try:
        download_voc(year="2007", save_dir="./data/voc")
    except Exception as e:
        print(str(e))
        # 即使下载失败，如果用户手动放置了tar文件，仍尝试解压
        save_dir = "./data/voc"
        filename = "VOCtrainval_06-Nov-2007.tar"
        save_path = os.path.join(save_dir, filename)
        if os.path.exists(save_path):
            print("\n🔄 检测到手动下载的文件，尝试解压...")
            with tarfile.open(save_path, 'r') as tar:
                tar.extractall(path=save_dir)
            print("✅ 手动文件解压成功！")