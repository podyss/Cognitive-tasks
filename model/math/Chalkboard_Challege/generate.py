import random
from PIL import Image, ImageDraw, ImageFont

# operators = ['+', '-', '*', '/']
operators = ['+', '-', '*']
def generate_expression(depth=2):
    if depth == 0:
        return str(random.randint(1, 10))
    left_expr = generate_expression(depth - 1)
    right_expr = generate_expression(depth - 1)
    operator = random.choice(operators)
    return f"({left_expr} {operator} {right_expr})"

def generate():
    img = Image.new('RGB', (256, 64), color='white')
    draw = ImageDraw.Draw(img)
    text = generate_expression()
    font = ImageFont.truetype('SimHei.ttf', size=20)
    box = draw.textbbox((0,0), text=text, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2
    draw.text((x, y), text, fill='black', font=font)
    # img.save('1.jpg')
    return img, text
