import cv2

trained_data_of_pedestrian = cv2.CascadeClassifier("haarcascade_fullbody.xml")
trained_data_of_car = cv2.CascadeClassifier("cars.xml")

video = cv2.VideoCapture("bike and car.mp4")# just change the value to 0 if you want to use webcam
while True:
	frame_read_successful, frame = video.read()

	if frame_read_successful:
		grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		break

	car_coordinates = trained_data_of_car.detectMultiScale(grayscaled_image)

	for (x, y, w, h) in car_coordinates:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

	pedestrian_coordinates = trained_data_of_pedestrian.detectMultiScale(grayscaled_image)

	for(x, y, w, h) in pedestrian_coordinates:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 3)

	cv2.imshow("Traffic tracker", frame)

	key = cv2.waitKey(1)

	if key == 81 or key == 113:
		break

video.release()





