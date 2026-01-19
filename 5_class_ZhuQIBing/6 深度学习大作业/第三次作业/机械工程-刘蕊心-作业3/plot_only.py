import pandas as pd
import matplotlib.pyplot as plt

# 读取已经保存的训练数据
csv_path = 'log_improved.csv'

try:
	# 读取CSV
	df = pd.read_csv(csv_path)
	print("成功读取数据，列名如下：")
	print(df.columns.tolist())
	
	# 自动识别列名（防止你之前的代码列名不统一）
	# 找 loss 相关的列
	train_loss_col = 'train_loss_all' if 'train_loss_all' in df.columns else 'train_loss'
	val_loss_col = 'val_loss_all' if 'val_loss_all' in df.columns else 'val_loss'
	
	# 找 acc 相关的列
	train_acc_col = 'train_acc_all' if 'train_acc_all' in df.columns else 'train_acc'
	val_acc_col = 'val_acc_all' if 'val_acc_all' in df.columns else 'val_acc'
	
	# 开始画图
	plt.figure(figsize=(12, 4))
	
	# 1. Loss 曲线
	plt.subplot(1, 2, 1)
	plt.plot(df['epoch'], df[train_loss_col], 'r-', label='Train Loss')
	plt.plot(df['epoch'], df[val_loss_col], 'b-', label='Val Loss')
	plt.title('Loss Curve')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	
	# 2. Accuracy 曲线
	plt.subplot(1, 2, 2)
	plt.plot(df['epoch'], df[train_acc_col], 'r-', label='Train Acc')
	plt.plot(df['epoch'], df[val_acc_col], 'b-', label='Val Acc')
	plt.title('Accuracy Curve')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid(True)
	
	# 保存并显示
	plt.savefig('training_result_fixed.png')
	print("✅ 图片已生成：training_result_fixed.png")
	plt.show()

except FileNotFoundError:
	print(f"❌ 没找到 {csv_path}，请确认你刚才的训练是否生成了这个文件。")
except Exception as e:
	print(f"❌ 发生错误: {e}")