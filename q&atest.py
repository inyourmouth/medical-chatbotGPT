import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast


# Thiết lập thông tin xác thực API
openai.api_key = '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'

# Đọc và xử lý bộ vector nhúng
def read_csv(filename):
    df = pd.read_csv('medical-embeddings.csv')
    embeddings = np.array(df['ada_v2'])
    sentences = df['CONTEXT'].tolist()
    
    return embeddings, sentences

# Tìm kiếm vector nhúng gần nhất
def find_nearest_embedding(question_embedding, embeddings, sentences):
    similarity_scores = cosine_similarity(question_embedding.reshape(1, -1), embeddings)
    nearest_index = np.argmax(similarity_scores)
    nearest_sentence = sentences[nearest_index]
    
    return nearest_sentence

# Hàm gửi câu hỏi và nhận câu trả lời từ GPT-3
def ask_question(question):
    response = openai.Completion.create(
        engine='text-embedding-ada-002', 
        model='text-davinci-002',
        prompt=question,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        api_version="2021-10-14"
    )
    
    if response.choices:
        return response.choices[0].text.strip()
    else:
        return "sorry, i can not answer."


# Hàm tương tác với người dùng
def chatbot_interaction():
    embeddings, sentences = read_csv('medical-embeddings.csv')

    while True: 
        user_input = input("Question: ")

        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break

        # Nhúng câu hỏi thành vector sử dụng API
        question_embedding = embed_question_with_api(user_input)

        print("question_embedding:", question_embedding)
        
        if question_embedding is not None:
            try:
                # Chuyển đổi câu trả lời từ API thành chuỗi
                response = ast.literal_eval(question_embedding)
                response_text = response['choices'][0]['text']

                
                # Nhúng câu trả lời thành vector
                answer_embedding = embed_question_with_api(response_text)

                print("answer_embedding:", answer_embedding)
                
                if answer_embedding is not None:
                    # Chuyển đổi chuỗi thành danh sách số thực
                    answer_embedding = ast.literal_eval(answer_embedding)

                    # Chuyển đổi danh sách thành mảng numpy
                    answer_embedding = np.array(answer_embedding)

                    # Tìm vector nhúng gần nhất
                    nearest_sentence = find_nearest_embedding(answer_embedding, embeddings, sentences)

                    # Gửi câu hỏi tới GPT-3
                    answer = ask_question(nearest_sentence)

                    print("Chatbot:", answer)
                else:
                    print("Chatbot: Sorry, I can't embed the answer.")
            except (ValueError, KeyError):
                print("Chatbot: Sorry, an error occurred while processing the response.")
        else:
            print("Chatbot: Sorry, I can't embed this question.")




def embed_question_with_api(question):

    # Địa chỉ API call để nhúng câu hỏi
    api_url = 'https://api.openai.com/v1/completions'

    # Tạo tiêu đề yêu cầu với khóa API
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}'
    }

    # Tạo payload hoặc các tham số cần thiết cho API call
    payload = {
        'model': 'text-davinci-002',
        'prompt': question,
        'max_tokens': 100,
        'n': 1,
        'stop': None,
        'temperature': 0.7,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }

    try:
        # Gửi API request để nhúng câu hỏi
        response = requests.post(api_url, headers=headers, json=payload)

        # In ra phản hồi từ API
        print("API response:", response.json())

        # Xử lý và trích xuất câu trả lời từ phản hồi của API
        if response.status_code == 200:
            answer = response.json()['choices'][0]['text']
            return answer.strip()
        else:
            print("API error:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("API connection error:", str(e))
        return None


# Gọi hàm tương tác với người dùng
chatbot_interaction()
