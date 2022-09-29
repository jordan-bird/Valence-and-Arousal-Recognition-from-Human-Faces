import argparse
import mediapipe as mp
import cv2
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to process", required=True)
parser.add_argument("--model", help="name of model to use (directory name)", required=True)
args = parser.parse_args()

model_name = args.model

img_url = args.image
print("Image: " + img_url)
image = cv2.imread(img_url)
height, width, channel = image.shape

def get_faces(verbose=False, extra_pixels=0):
	mp_face_detection = mp.solutions.face_detection
	mp_drawing = mp.solutions.drawing_utils

	with mp_face_detection.FaceDetection(
		model_selection=1, min_detection_confidence=0.5) as face_detection:
		results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	print(results)

	faces = []
	for detection in results.detections:
		bbox = detection.location_data.relative_bounding_box
		bbox_points = {
			"xmin" : int(bbox.xmin * width),
			"ymin" : int(bbox.ymin * height),
			"xmax" : int(bbox.width * width + bbox.xmin * width),
			"ymax" : int(bbox.height * height + bbox.ymin * height)
		}
		
		xmin = int(bbox_points["xmin"]) + extra_pixels
		xmax = int(bbox_points["xmax"]) + extra_pixels
		ymin = int(bbox_points["ymin"]) + extra_pixels
		ymax = int(bbox_points["ymax"]) + extra_pixels
		
		
		face = image[ymin:ymax, xmin:xmax]
		faces.append(face)
		
		if(verbose):
			cv2.imshow("face", face)
			cv2.waitKey()
			print(faces)
		
	return faces


def predict(faces, verbose=False):
	predictions = []
	val_aro_model = tf.keras.models.load_model(model_name)
	
	for count, face in enumerate(faces):
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		
		face = cv2.resize(face, (128,128), interpolation= cv2.INTER_LANCZOS4)
			
		pred = val_aro_model.predict(face.reshape(1,128,128,1))
		valence = pred[0][0][0]
		arousal = pred[1][0][0]
		
		if(verbose):
			print("Face ID: " + str(count))
			print("Valence: " + str(valence))
			print("Arousal: " + str(arousal))
			cv2.imshow("face", face)
			cv2.waitKey()
		
		predictions.append([valence, arousal])
	
	return predictions

faces = get_faces(verbose=False)
print("Faces detected: " + str(len(faces)))
predictions = predict(faces, verbose=True)

print(predictions)

