# TryOnNow

👕 Virtual Try-On AR Shirt Experience

This project is a virtual Augmented Reality (AR) experience that lets users try on shirts from home using a webcam. By uploading a single 2D shirt image, the system generates a realistic 3D mesh and overlays it onto the human body in real time — giving the illusion that the user is actually wearing the shirt.

🎯 Goal

• Simulate the in-store shirt try-on experience at home
• Provide a realistic, live view of how a shirt fits and looks
• Help users make better clothing decisions before purchase
• Merge computer vision, machine learning, and real-time rendering into one seamless experience

🛠️ Technologies Used

Category	Tools / Frameworks
Backend	Python, Flask
Computer Vision	OpenCV, MediaPipe, YOLOv8
3D Rendering	Pyrender, Trimesh, Three.js
Machine Learning	SAM (Segment Anything), K-means clustering, LBP
Frontend	HTML5, CSS3, JavaScript

🔁 Full Working Flow

👤 User accesses the Flask web interface and uploads a 2D shirt image

🎯 YOLOv8 model detects the shirt area from the uploaded image

🧠 Garment features (edges, color, pattern) are extracted using:
• SAM for precise segmentation
• LBP for texture features
• K-means clustering for color zone separation

🎥 The webcam turns on, and MediaPipe tracks real-time body pose (shoulders, torso, etc.)

🧵 The extracted shirt features are passed into the 3D mesh generator

🧊 Trimesh and Pyrender convert the 2D image into a dynamic 3D mesh

🌐 Three.js renders this mesh inside the HTML canvas, matching the user’s pose

🪄 The shirt is "attached" to the user’s upper body and moves with them

📱 The result is a live, AR-based virtual try-on experience — as if the user is wearing the shirt

🪢 Key Features

• Real-time virtual try-on via webcam
• Accurate pose detection using MediaPipe
• YOLO-based shirt region detection
• Converts any 2D shirt image into a wearable 3D mesh
• Mesh aligns and sticks to user’s body based on movement
• Lightweight, browser-based, and mobile-responsive
• Built with modular components for easy updates

📌 Example Use Case

Imagine you're browsing an online store and want to see how a particular shirt looks on you before buying. This system allows you to upload the shirt image and try it on instantly using your webcam — without any physical fitting room.

🧪 Future Enhancements

• Support for full-body garments (pants, jackets, dresses)
• Better fabric simulation (wrinkles, folds, draping)
• Size fitting based on body proportions
• Integration with online clothing stores

