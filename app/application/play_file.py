import app.config as cf

from app.application.translate import translate

if __name__ == "__main__":
    sentence = input("Enter the sentence to translate : ")
    predicted_sentence = translate(sentence)

    print(f"Predicted translation : {predicted_sentence}")

