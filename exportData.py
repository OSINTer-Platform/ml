import json
from OSINTmodules import *

configOptions = OSINTconfig.backendConfig()

esClient = OSINTelastic.returnArticleDBConn(configOptions)

def main(outputPath):
    articles = esClient.searchDocuments({"limit" : 10000})["documents"]

    data = {"labels" : [], "numberLabels" : [], "texts" : [], "labelList" : []}

    for currentArticle in articles:
        data["labels"].append(currentArticle.title)
        data["texts"].append(currentArticle.content)

    labelList = list(set(data["labels"]))
    labelList = { labelList[i] : i for i in range(len(labelList))}

    for label in data["labels"]:
        data["numberLabels"].append(labelList[label])

    data["labelList"] = list(labelList)

    for fileName in ["numberLabels", "texts", "labelList"]:
        with open(f"{outputPath}/{fileName}.txt", "w") as f:
            json.dump(data[fileName], f)


if __name__ == "__main__":
    main(input("Enter path for data output files: "))
