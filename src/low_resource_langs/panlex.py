import pandas as pd
import requests
import time
import json
from tqdm import tqdm

CHEROKEE_LANG_CODE = 114
BASE_URL = "http://api.panlex.org/v2/expr"

if __name__ == "__main__":
    expr_df = pd.read_csv("./panlex-20230801-csv/expr.csv")
    chr_words = expr_df.loc[expr_df["langvar"] == CHEROKEE_LANG_CODE]
    chr_forms = chr_words["txt"]
    chr_ids = chr_words["id"]
    
    # TODO: for some reason I can't find the expressions here in the meaning table.
    # their api allows fetching translations though...
    headers = {
        "Content-Type": "application/json"
    }
    
    en_translations = []
    for cid, cword in tqdm(zip(chr_ids, chr_forms), total=len(chr_ids)):
        data = {
            "uid": "eng-000",
            "trans_expr": cid
        }

        response = requests.post(BASE_URL, headers=headers, data=json.dumps(data))
        result = response.json()["result"]
        all_translations = ",".join([x["txt"] for x in result])
        en_translations.append(all_translations)
        print(f"{cword} -> {all_translations} [{response.status_code}]")
        time.sleep(1)

    df = pd.DataFrame({"chr": chr_forms, "en": en_translations})
    df.to_csv("./data/cherokee-panlex/translations.csv", index=False)
