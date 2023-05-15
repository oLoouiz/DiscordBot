import cv2
import tensorflow as tf

#loaded_model = tf.keras.saving.load_model("modelo\modelo_ants-bees.h5")
loaded_model = tf.keras.saving.load_model("modelo/retreinado.h5")

test_img = cv2.imread('imagem test/abelha.png')
janela_abelha = 'test_img'
cv2.imshow(janela_abelha,test_img)

test_img.shape
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
cv2.waitKey(0)

print(loaded_model.predict(test_input))

test_img = cv2.imread('imagem test/formiga.png')
janela_formiga = 'formiga_img'
cv2.imshow(janela_formiga,test_img)

test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
print(loaded_model.predict(test_input))

cv2.waitKey(0)
cv2.destroyAllWindows()