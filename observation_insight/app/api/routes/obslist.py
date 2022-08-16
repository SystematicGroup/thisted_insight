from fastapi import APIRouter
from starlette.requests import Request
from fastapi import Response, status
import csv
from observation_insight.app.core.config import args

router = APIRouter()

@router.get("/observationlist", name="Give a list of possible observation types")
def give_list():
    file = args['obslist_dir']+'/Observation_types_v2.csv'
    rows = []
    with open(file, 'r') as filedat:
        csvreader = csv.reader(filedat)
        next(csvreader) #Skip the first line, witch is a header.
        for row in csvreader:
            rows += row
    data_list = {'result': rows}

    return data_list