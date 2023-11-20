from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


def pyramid_fusion(image1_path, image2_path, output_path):
    # 加载图像
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 将PIL图像转换为张量
    to_tensor = ToTensor()
    image1 = to_tensor(image1)
    image2 = to_tensor(image2)

    # 平均法
    fused_image = (image1 + image2) / 2

    # 将张量转换回PIL图像
    to_pil_image = ToPILImage()
    fused_image = to_pil_image(fused_image)

    # 保存融合后的图像
    fused_image.save(output_path)


pyramid_fusion("images/IV_images/IR1.png", "images/IV_images/VIS1.png", "fuse_result/result.jpg")
