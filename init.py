import cv2 as cv
from label_image import *

model_file = './tf_files/mobilenet_1.0_224_graph.pb'
label_file = './tf_files/mobilenet_1.0_224_labels.txt'
input_layer = 'input'
output_layer = 'final_result'

input_name = 'import/' + input_layer
output_name = 'import/' + output_layer

graph = load_graph(model_file)
input_op = graph.get_operation_by_name(input_name)
output_op = graph.get_operation_by_name(output_name)
labels = load_labels(label_file)

font = cv.FONT_HERSHEY_COMPLEX

def predict_from_cam():
    vid = cv.VideoCapture(1)

    while True:
        _, frame = vid.read()
        preds = prediction(frame, graph, labels, input_op, output_op)

        for i, pred in enumerate(preds):
            cv.putText(frame, str(pred[0]) + ' : ' + str(pred[1]),
                       (10, 80+(20*i)), font, 0.7, (0, 0, 255))

        cv.imshow('Flower Classifier Cam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    predict_from_cam()
