import cv2
import numpy as np
import requests
import io
import json

img = cv2.imread("Annotation 2020-04-25 215317.jpg")  # cv2.imread() method loads an image from the specified file
height, width, _ = img.shape                         # to take only selected part of the  image

# Cutting image
#roi = img[0: height, 400: width]
roi = img                                             # we are storing the image in roi variable so the real image is not affected

# Ocr
url_api = "https://api.ocr.space/parse/image"         # with this we are parsing the roi image
_, compressedimage = cv2.imencode(".jpeg", roi, [1, 90]) # after parsing, cv2.imencode() converts data into image format
file_bytes = io.BytesIO(compressedimage)

result = requests.post(url_api,                         # now the parsed roi image is stored in result in jpeg format
              files = {"screenshot.jpeg": file_bytes},
              data = {"apikey": "e6ef27968188957",
                      "language": "eng"})

result = result.content.decode()                        # here it does the Segmentation and Feature extraction process
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")    # here in this the parsed text from segmentation process is stored in text_detection
print(text_detected)

cv2.imshow("roi", roi)      # to show the image read by cv2
cv2.imshow("Img", img)      # to show the input Image
cv2.waitKey(0)              # used to hold the image until we press a key
cv2.destroyAllWindows()