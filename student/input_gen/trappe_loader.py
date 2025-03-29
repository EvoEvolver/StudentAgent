import json
import requests


def request_by_post(url, payload=None):
    # Send a POST request
    response = requests.post(url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def download_properties(molecule_id: int):
    payload = {"molecule_id": molecule_id}
    csv_string = request_by_post("http://trappe.oit.umn.edu/scripts/download_properties.php", payload)
    return csv_string

def download_parameters(molecule_id: int):
    payload = {"molecule_id": molecule_id}
    csv_string = request_by_post("http://trappe.oit.umn.edu/scripts/download_parameters.php", payload)
    return csv_string


def get_molecule_list():
    # URL to scrape
    url = "http://trappe.oit.umn.edu/scripts/search_select.php"
    # check if the data is already downloaded
    try:
        with open("trappe_molecule_list.json") as f:
            return json.load(f)["search"]
    except FileNotFoundError:
        pass

    res_string = request_by_post(url)
    res_dict = json.loads(res_string)
    # save the data to a file
    with open("trappe_molecule_list.json", "w") as f:
        json.dump(res_dict, f)

    return res_dict["search"]

def get_molecule_list_parsable():
    id_list = {int(i["molecule_ID"]) for i in get_molecule_list()}
    return list(id_list)


# TODO

def get_flexible_molecule_input(bond, bend):
    res = []
    for b in bond[1:]:
        res.append(b[1].replace("-", " "))
        res.append("HARMONIC_BOND")

    return " ".join(res)



if __name__ == '__main__':
    #atoms, bond_lengths, bond_angles = get_trappe_parameters(18)
    #print(atoms, bond_lengths, bond_angles)
    #atoms, bond_lengths, bond_angles = get_trappe_parameters(18)
    #print(get_flexible_molecule_input(bond_lengths, bond_angles))

    generate_molecule_def(molecule_id=3, output_file="molecule.def")