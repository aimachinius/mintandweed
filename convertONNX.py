from ultralytics import YOLO
model = YOLO("best.pt")

model.export(format="onnx",dynamic=True, opset=12, task='segmentation',imgsz=512)
# import onnx
# onnx_model = onnx.load("best.onnx")
# onnx.checker.check_model(onnx_model)
# print("✅ ONNX model is valid!")
