import time
import cv2
import scipy.special
from enum import Enum
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except Exception as e:
    from tensorflow.lite.python.interpreter import Interpreter
    print("程式異常:", e)

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE):

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path)

	def initialize_model(self, model_path):

		self.interpreter = Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg, self.output_details)
		#print("lanes_detected:", self.lanes_detected)
		#print("lanes_points:", self.lanes_points)

		# # Draw depth image
		#visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)
		visualization_img = type(self).draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape
	
		# Input values should be from -1 to 1 with a size of 288 x 800 pixels
		img_input = cv2.resize(img, (self.input_width, self.input_height)).astype(np.float32)
	
		if self.input_dtype == np.float32:
			# Scale input pixel values to -1 to 1
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			img_input = img_input.astype(np.float32)
			img_input = ((img_input / 255.0 - mean) / std).astype(np.float32)
		else:
			img_input = img_input.astype(np.uint8)
	
		img_input = img_input[np.newaxis, :, :, :]
		return img_input

	def getModel_input_details(self):
		self.input_details = self.interpreter.get_input_details()
		input_shape = self.input_details[0]['shape']
		self.input_dtype = self.input_details[0]['dtype']
		self.input_height = input_shape[1]
		self.input_width = input_shape[2]
		self.channels = input_shape[3]

	def getModel_output_details(self):
		self.output_details = self.interpreter.get_output_details()
		output_shape = self.output_details[0]['shape']
		self.num_anchors = output_shape[1]
		self.num_lanes = output_shape[2]	
		self.num_points = output_shape[3]

	def inference(self, input_tensor):
		# Peform inference
		self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
		self.interpreter.invoke()
		output = self.interpreter.get_tensor(self.output_details[0]['index'])

		output = output.reshape(self.num_anchors, self.num_lanes, self.num_points)

		#print("output shape:", output.shape)
		#print("output sample:", output.flatten()[:20])
		#print("output has NaN:", np.isnan(output).any())
		#print("output has Inf:", np.isinf(output).any())
		#print("output dtype:", output.dtype)
		return output


	@staticmethod
	def process_output(output, cfg, output_details=None):
		# 量化模型反量化
		if output.dtype == np.uint8 or output.dtype == np.int8:
			if output_details is not None:
				scale = output_details[0]['quantization'][0]
				zero_point = output_details[0]['quantization'][1]
			else:
				scale = 1.0
				zero_point = 0
			output = (output.astype(np.float32) - zero_point) * scale
		processed_output = output[:, ::-1, :]

		#print("processed_output (before softmax) min/max:", np.nanmin(processed_output), np.nanmax(processed_output))
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		# idx = np.arange(cfg.griding_num) + 1
		# idx = idx.reshape(-1, 1, 1)
		num_grids = prob.shape[0]
		idx = np.arange(1, num_grids + 1).reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc

		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lane_points_mat = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			#print("processed_output shape:", processed_output.shape)
			#print("processed_output sample:", processed_output[:10, :])

			lane_points_mat.append(lane_points)
		# return np.array(lane_points_mat), np.array(lanes_detected) BBB
			max_len = max(len(lane) for lane in lane_points_mat)
			lane_points_mat_pad = [lane + [(-1, -1)] * (max_len - len(lane)) for lane in lane_points_mat]
		return np.array(lane_points_mat_pad), np.array(lanes_detected)

	@staticmethod
	def draw_lanes(input_img, lane_points_mat, lanes_detected, cfg, draw_points=True):
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)
	
		# 自動選擇最左和最右的有效車道來填色
		valid_lanes = [i for i, detected in enumerate(lanes_detected) if detected]
		if lanes_detected[1] and lanes_detected[2]:
			left_lane = np.array([pt for pt in lane_points_mat[1] if pt[0] >= 0 and pt[1] >= 0], dtype=np.int32)
			right_lane = np.array([pt for pt in lane_points_mat[2] if pt[0] >= 0 and pt[1] >= 0], dtype=np.int32)
			if left_lane.shape[0] > 1 and right_lane.shape[0] > 1:
				# 用左車道全部點 + 右車道全部點（反向）組成多邊形
				pts = np.vstack((left_lane, np.flipud(right_lane)))
				mask = visualization_img.copy()
				cv2.fillPoly(mask, [pts], color=(255, 200, 255))
				visualization_img = cv2.addWeighted(visualization_img, 0.7, mask, 0.3, 0)

		# 畫所有偵測到的車道線
		for lane_num, lane_points in enumerate(lane_points_mat):
			pts = np.array([pt for pt in lane_points if pt[0] >= 0 and pt[1] >= 0], dtype=np.int32)
			if pts.shape[0] > 1:
				cv2.polylines(visualization_img, [pts], isClosed=False, color=lane_colors[lane_num % len(lane_colors)], thickness=2)

		# 畫所有偵測到的車道點
		if draw_points:
			for lane_num, lane_points in enumerate(lane_points_mat):
				for lane_point in lane_points:
					if lane_point[0] >= 0 and lane_point[1] >= 0:
						cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num % len(lane_colors)], -1)

		return visualization_img

