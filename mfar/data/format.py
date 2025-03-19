import json
import numpy as np
from typing import Any, Tuple, List


# TODO: format_documents is redundant w/ format_dict, maybe can make the two functions recursive...
def format_documents(documents, field_name, dataset_name):
    """
    Formats the documents such that we only include the 'type' key we care about.

    Parameters:
        documents: the set of documents to format
        field: the Field object specifying what to extract

    Returns: the new document set

    # For amazon: title = str, brand = str, description = list,
    # review = list of dicts (with the same keys), also_buy = list, also_view = list
    """
    if field_name == "single":
        all_docs = [format_stark(document, dataset_name) for document in documents]
        return all_docs

    ids_list = [doc[0] for doc in documents]
    docs_list = []

    for doc in documents:
        if field_name in doc[1]:
            string = ""
            value = doc[1][field_name]
            # Three options: str/int, list, list of dicts (for Amazon)
            if isinstance(value, str):
                string += value
            elif isinstance(value, int) or isinstance(value, float):
                string += str(value)
            elif isinstance(value, list):
                # Check if it's empty, first
                if len(value) == 0:
                    string += ""
                elif isinstance(value[0], dict):
                    converted_dict_strings = []
                    for item in value:
                        filtered_dict = {k: v for k, v in item.items()
                                        if k not in ["reviewerID", "style", "verified", "overall", "reviewTime", "vote", "questionType", "answerType", "answerTime"]}
                        new_string = "\n".join([f"{k}: {str(v)}" for k, v in filtered_dict.items()])
                        converted_dict_strings.append(new_string)
                    string += "\n".join(converted_dict_strings)
                elif isinstance(value[0], list):
                    raise NotImplementedError("Nested list not supported!")
                else:
                    string += ", ".join(value)
            elif value is None:
                string += ""
            else:
                string += format_dict(value)

            docs_list.append(string)
        else:
            docs_list.append("")

    return list(zip(ids_list, docs_list))


def format_dict(item_dict):
    """
    Formats dict-specific items into a string

    Parameters:
        item: the dict to format

    Returns: the formatted string
    """

    # For Prime, we can have dict w/ values of str, list, and dict... (list might be nested too)
    all_strings = []
    for key in item_dict:
        value = item_dict[key]
        if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
            all_strings.append(f"{key}: {value}")
        elif isinstance(value, list):
            if len(value) == 0:
                all_strings.append(f"{key}: ")
            elif isinstance(value[0], dict):
                fields_to_text = {}
                for item in value:
                    for k, v in item.items():
                        if k not in fields_to_text:
                            fields_to_text[k] = []
                        if isinstance(v, dict):
                            fields_to_text[k].extend(list(v.values()))
                        else:
                            fields_to_text[k].append(v)
                string = ""
                for k in fields_to_text:
                    items = fields_to_text[k]
                    items = [str(item) for item in items]
                    string += f"{k}: {', '.join(items)}; "
                all_strings.append(string)
            elif isinstance(value[0], list):
                raise NotImplementedError("Nested list not supported!")
            else:
                items = ", ".join(value)
                all_strings.append(f"{key}: {items}")
        elif isinstance(value, dict):
            items = [f"{k}: {value[k]}" for k in value]
            all_strings.append(", ".join(items))
        else:
            all_strings.append(", ".join(value))

    return "; ".join(all_strings)


def format_stark(data: Tuple[str, Any], dataset_name: str) -> Tuple[str, Any]:
    """
    Formats the information in the original STaRK format (as close as possible).
    Formatting is inspired by the STaRK repo, https://github.com/snap-stanford/stark

    Parameters:
    data: Dict[str, Any] -- the data to format
    dataset: the specific dataset

    Returns: a tuple containing the doc_id and data
    """

    _id, _data = data
    if dataset_name == "amazon":
        doc = format_amazon(_data)
    elif dataset_name == "mag":
        doc = format_mag(_data)
    elif dataset_name == "prime":
        doc = format_prime(_data)
    elif dataset_name == "whatsthatbook" or dataset_name == "tomt":
        doc = format_books(_data)
    else:
        raise ValueError("Select a valid STaRK dataset!")

    return (_id, doc)

def format_amazon(data: Tuple[str, Any]) -> List[str]:
    """
    Formats the Amazon (STaRK) dataset

    Parameters:
    data: Dict[str, Any] -- the data to format

    Returns: the string format of the doc
    """
    doc = f'- product: {data["title"]}\n'
    if 'brand' in data:
        doc += f'- brand: {data["brand"]}\n'

    if 'description' in data:
        description = " ".join(data['description']).strip(" ")
        if description:
            doc += f'- description: {description}\n'

    feature_text = '- features: \n'
    if 'feature' in data:
        for feature_idx, feature in enumerate(data['feature']):
            if feature and 'asin' not in feature.lower():
                feature_text += f'#{feature_idx + 1}: {feature}\n'
    else:
        feature_text = ''

    if 'review' in data:
        review_text = '- reviews: \n'
        for i, review in enumerate(data['review']):
            review_text += f'#{i + 1}:\nsummary: {review["summary"]}\ntext: "{review["reviewText"]}"\n'
    else:
        review_text = ''

    if 'qa' in data:
        qa_text = '- QA: \n'
        for qa_idx, qa in enumerate(data['qa']):
            qa_text += f'#{qa_idx + 1}:\nquestion: {qa["question"]}\nanswer: {qa["answer"]}\n'
    else:
        qa_text = ''

    doc += feature_text + review_text + qa_text
    doc += get_amazon_rel_info(data)

    return doc

def get_amazon_rel_info(data: Tuple[str, Any], n_rel: int = -1):
    """
    Gets the relations for the Amazon (STaRK) dataset.

    Parameters:
    data: Dict[str, Any] -- the data to format
    n_rel: int -- the number of relations to include, -1 if all

    Returns: the string format of the doc + rels
    """

    doc = ''
    if 'also_buy' in data:
        str_also_buy = [f"#{idx + 1}: " + i + '\n' for idx, i in enumerate(data['also_buy'])]
    if 'also_view' in data:
        str_also_view = [f"#{idx + 1}: " + i + '\n' for idx, i in enumerate(data['also_view'])]

    if n_rel > 0:
        str_also_buy = str_also_buy[:n_rel]
        str_also_view = str_also_view[:n_rel]

    if not str_also_buy:
        str_also_buy = ''
    if not str_also_view:
        str_also_view = ''

    str_also_buy = ''.join(str_also_buy)
    str_also_view = ''.join(str_also_view)

    if str_also_buy:
        doc += f'  products also purchased: \n{str_also_buy}'
    if str_also_view:
        doc += f'  products also viewed: \n{str_also_view}'
    if 'brand' in data:
        doc += f'  brand: {data["brand"]}\n'

    if doc:
        return ' - relations:\n' + doc
    else:
        return ''


def format_mag(data: Tuple[str, Any]) -> List[str]:
    """
    Formats the Mag (STaRK) dataset

    Parameters:
    data: Dict[str, Any] -- the data to format

    Returns: the string format of the doc
    """
    if data['type'] == 'paper':
        doc = f' - paper title: {data["title"]}\n'
        doc += ' - abstract: ' + data["abstract"].replace('\r', '').rstrip('\n') + '\n'

    doc += get_mag_rel_info(data)

    return doc

def get_mag_rel_info(data: Tuple[str, Any]):
    """
    Gets the relations for the Mag (STaRK) dataset.

    Parameters:
    data: Dict[str, Any] -- the data to format

    Returns: the string format of the doc + rels
    """

    doc = ''
    str_cites, str_references, str_affiliated = '', '', ''
    if "paper___cites___paper" in data:
        str_cites = [f'\"{i}\"' for i in data["paper___cites___paper"]]
        str_cites = 'paper cites paper: (' + ', '.join(str_cites) + ')'
    if "paper___has_topic___field_of_study" in data:
        str_references = 'paper has_topic field_of_study: (' + ', '.join(data["paper___has_topic___field_of_study"]) + ')'
    if "author___affiliated_with___institution" in data:
        info = data["author___affiliated_with___institution"]
        all_authors = []
        for author in info:
            institutions = info[author]
            all_inst = '(' + ', '.join(institutions) + ')'
            all_authors.append(author + " " + all_inst)

        str_affiliated = '(' + ', '.join(all_authors) + ')'

    doc = ',\n'.join(filter(None, [str_cites, str_references, str_affiliated]))

    if doc:
        return ' - relations:\n\n' + doc
    else:
        return ''

def format_prime(data: Tuple[str, Any]) -> List[str]:
    """
    Formats the Prime (STaRK) dataset

    Parameters:
    data: Dict[str, Any] -- the data to format

    Returns: the string format of the doc
    """
    if "name" not in data:
        print(f"format_prime Error: \"name\" not found in {data}. This should be required.")
        return ""
    doc = f'- name: {data["name"]}\n'
    doc += f'- type: {data["type"]}\n'
    doc += f'- source: {data["source"]}\n'
    gene_protein_text_explain = {
        'name': 'gene name',
        'type_of_gene': 'gene types',
        'alias': 'other gene names',
        'other_names': 'extended other gene names',
        'genomic_pos': 'genomic position',
        'generif': 'PubMed text',
        'interpro': 'protein family and classification information',
        'summary': 'protein summary text'
    }

    feature_text = f'- details: \n'
    feature_cnt = 0
    if 'details' in data:
        for key, value in data['details'].items():
            if str(value) in ['', 'nan'] or key.startswith('_') or '_id' in key:
                continue
            if data['type'] == 'gene/protein' and key in gene_protein_text_explain.keys():
                if 'interpro' in key:
                    if isinstance(value, dict):
                        value = [value]
                        value = [v['desc'] for v in value]
                if 'generif' in key:
                    value = '; '.join([v['text'] for v in value])
                    value = ' '.join(value.split(' ')[:50000])
                if 'genomic_pos' in key:
                    if isinstance(value, list):
                        value = value[0]
                feature_text += f'  - {key} ({gene_protein_text_explain[key]}): {value}\n'
                feature_cnt += 1
            else:
                feature_text += f'  - {key}: {value}\n'
                feature_cnt += 1
    if feature_cnt == 0:
        feature_text = ''

    doc += feature_text
    doc += get_prime_rel_info(data)

    return doc

def get_prime_rel_info(data: Tuple[str, Any]):
    """
    Gets the relations for the Prime (STaRK) dataset.

    Parameters:
    data: Dict[str, Any] -- the data to format
    n_rel: int -- the number of relations to include, -1 if all

    Returns: the string format of the doc + rels
    """

    relation_types = [
        'ppi', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication',
        'indication', 'off-label use', 'synergistic interaction', 'associated with',
        'parent-child', 'phenotype absent', 'phenotype present', 'side effect',
        'interacts with', 'linked to', 'expression present', 'expression absent'
    ]

    all_items = []
    for relation in relation_types:
        if relation in data:
            item = f"  {relation.replace(' ', '_')}: " + "{"
            relation_items = []
            for key in data[relation]:
                rel_item = f"{key.replace(' ', '_')}: "
                rel_item += '(' + ', '.join(data[relation][key]) + ')'
                relation_items.append(rel_item)
            item += ', '.join(relation_items)
            item += "}"
            all_items.append(item)

    doc = '\n'.join(filter(None, all_items))

    if doc:
        return ' - relations:\n' + doc
    else:
        return ''

def format_books(data: Tuple[str, Any]) -> List[str]:
    """
    Formats the Prime (STaRK) dataset

    Parameters:
    data: Dict[str, Any] -- the data to format

    Returns: the string format of the doc
    """

    doc = f'- title: {data["title"]}\n' if "title" in data else ''
    doc += f'- author: {data["author"]}\n' if "author" in data else ''
    doc += f'- author url: {data["author_url"]}\n' if "author_url" in data else ''
    doc += f'- description: {data["description"]}\n' if "description" in data else ''
    doc += f'- isbn: {data["isbn"]}\n' if "isbn" in data else ''

    if "parsed_dates" in data:
        all_dates = []
        if data['parsed_dates'] != None:
            for dates in data['parsed_dates']:
                if dates != None:
                    all_dates.append(dates)

        if len(all_dates) != 0:
            doc += f'- parsed dates: {", ".join(all_dates)}\n'

    doc += f'- image link: {data["image_link"]}\n' if "image_link" in data else ''
    doc += f'- number of ratings: {data["num_ratings"]}\n' if "num_ratings" in data else ''
    doc += f'- number of reviews: {data["num_reviews"]}\n' if "num_reviews" in data else ''
    if "genres" in data:
        if len(data['genres']) != 0:
            genres = ', '.join(data["genres"])
            doc += f'- genres: {genres}\n'
    doc += f'- id: {data["id"]}' if "id" in data else ''

    return doc

# From: https://stackoverflow.com/questions/58408054/typeerror-object-of-type-bool-is-not-json-serializable
class CustomJSONizer(json.JSONEncoder):
    """
    Used mainly to deal with weird type conversions (namely, numpy bools)
    """
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)