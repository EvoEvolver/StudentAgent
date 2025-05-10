import json
import requests
from ...utils import request_by_post

def download_properties(molecule_id: int):
    if molecule_id == 1: # methane has incomplete property data for id=1
        molecule_id = 164

    payload = {"molecule_id": molecule_id}
    csv_string = request_by_post("http://trappe.oit.umn.edu/scripts/download_properties.php", payload)
    return csv_string

def download_parameters(molecule_id: int):
    payload = {"molecule_id": molecule_id}
    csv_string = request_by_post("http://trappe.oit.umn.edu/scripts/download_parameters.php", payload)
    return csv_string

