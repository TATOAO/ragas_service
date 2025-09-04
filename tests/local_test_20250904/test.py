

# parsing excel file
import pandas as pd

df = pd.read_excel('用友测试集100条-4B打分.xlsx')

print(df.head())


# parsing csv file


# cd tests/local_test_20250904
# python test.py


# create dataset    
import requests

headers = {
    "Authorization": "Bearer test-api-key"
}
def create_dataset():
    url = "http://localhost:8887/api/v1/datasets/"

    data = {
        "name": "用友测试集100条-4B打分",
        "description": "用友测试集100条-4B打分",
        "sample_type": "single_turn",
        "metadata": {"source": "test", "version": "1.0"}
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    assert response.status_code == 200

def insert_samples():
    url = "http://localhost:8887/api/v1/datasets/datasetname/用友测试集100条-4B打分/samples"
    for index, row in df.iterrows():
        data = {
            "user_input": row["查询内容"],
            "retrieved_contexts": [row["原始上下文"]],
        }
        response = requests.post(url, headers=headers, json=data)
        print(response.json())

def delete_dataset():
    url = "http://localhost:8887/api/v1/datasets/datasetname/用友测试集100条-4B打分"
    response = requests.delete(url, headers=headers)
    print(response.json())

# python test.py
if __name__ == "__main__":
    # create_dataset()
    # insert_samples()
    delete_dataset()