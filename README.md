🎨 NEURAL-STYLE-TRANSFER
COMPANY: CODTECH IT SOLUTIONS PVT. LTD

NAME: Limbani Uttam Bharatbhai

INTERN ID: CT04DG2987

DOMAIN: Artificial Intelligence

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

This project implements a Neural Style Transfer (NST) model using TensorFlow and the pretrained VGG19 network, allowing you to blend the artistic style of one image (style image) with the content of another (content image). The outcome is a striking new image that appears as if the original photo was painted by a famous artist — showcasing the creative power of deep learning.

📜 Project Description
Neural Style Transfer is a computer vision technique that applies convolutional neural networks (CNNs) to combine:

The content of one image (e.g., a photograph)

With the style of another (e.g., a painting)

This implementation uses:

VGG19 to extract and manipulate content/style features

TensorFlow 2.x for the training loop and optimization

Gram matrices to represent the style texture

Custom loss functions to balance content and style influences

🧠 How It Works
🔧 Preprocessing:
Load and resize content and style images

Convert to tensors and normalize using VGG19 preprocessing

📤 Feature Extraction:
Extract deeper layers for content features

Extract shallower layers for style features

Compute Gram matrices from style features

📉 Loss Calculation:
Content Loss: Difference between generated and content image features

Style Loss: Difference between generated and style Gram matrices

Total Loss = (Content Loss × content weight) + (Style Loss × style weight)

🚀 Optimization:
Uses Adam optimizer with gradient descent

Iteratively updates the target image to minimize total loss

🚀 Features
✅ Utilizes pre-trained VGG19 model from Keras
✅ Compatible with TensorFlow 2.x
✅ Supports real-time image transformation visualization
✅ Adjustable weights for content/style emphasis
✅ Saves the final stylized output image

🖼️ Input
Content Image

![pexels-souvenirpixels-414612](https://github.com/user-attachments/assets/82807c84-1368-44df-8264-142e57d2f63f)

Style Image

![pexels-scottwebb-430207](https://github.com/user-attachments/assets/eb0e8bdc-1065-464f-9a42-b1b63c042b60)

🎨 Output
Output Image: Figure_1

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/16604810-580d-4373-86fa-a9ea8f0f306e" />

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/d00fab1b-f2d7-41e9-a0ea-b46083289b6f" />

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/a0ab10c8-c488-4380-802c-06e1af5c34a1" />

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/4a5710d7-acab-4c7e-abed-5310d63b3f55" />

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/a1561935-fe35-4abf-8e10-b5ad705fdee7" />

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/b84d2546-f96d-48a5-a617-3e6c3c4feddf" />

