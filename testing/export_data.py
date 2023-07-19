import json
from modules import config, elastic

config_options = config.BackendConfig()

def main(outputPath):
    articles = config_options.es_article_client.query_documents(elastic.SearchQuery(limit = 10000, complete=True))["documents"]

    data = {"labels" : [], "number_labels" : [], "texts" : [], "label_list" : []}

    for current_article in articles:
        data["labels"].append(current_article.title)
        data["texts"].append(current_article.content)

    label_list = list(set(data["labels"]))
    label_list = { label_list[i] : i for i in range(len(label_list))}

    for label in data["labels"]:
        data["number_labels"].append(label_list[label])

    data["label_list"] = list(label_list)

    for filename in ["number_labels", "texts", "label_list"]:
        with open(f"{outputPath}/{filename}.txt", "w") as f:
            json.dump(data[filename], f)


if __name__ == "__main__":
    main(input("Enter path for data output files: "))
