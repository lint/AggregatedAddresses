import shutil
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon
from shapely.affinity import translate
from shapely.geometry import mapping
import json
import os
from concurrent.futures import ThreadPoolExecutor

# define data directories
data_in_dir = 'in'
data_feather_dir = 'feather'
data_out_dir = 'data'

states = [ 'ca', 'vt', 'va', 'pr', 'ak', 'sd', 'sc', 'ut', 'ga', 'ms', 'mt', 'mo', 'ma', 'ky', 'al', 'nh', 'mn', 'mi', 'ok', 'in', 'co', 'ia', 'ct', 'fl', 'wv', 'ri', 'wy', 'tx', 'pa', 'nc', 'nd', 'nm', 'nj', 'me', 'ar', 'nv', 'dc', 'md', 'ks', 'ne', 'hi', 'de', 'az', 'ny', 'id', 'oh', 'or', 'il', 'la', 'wi', 'wa', 'tn']

# load bounding boxes for each us state
with open("us_bounding_boxes.json", 'r') as f:
    boxesData = json.loads(f.read())

# return the bounding box for a given state
def boundsForState(state):
    if state.upper() in boxesData:
        return boxesData[state.upper()]
    else:
        return [-1.67, -1.93, 164.09, 72.24]

# create an empty hex grid within the given bounds
def create_hex_grid(bounds, hex_size):
    minx, miny, maxx, maxy = bounds
    hex_width = hex_size
    hex_height = hex_size * (3 ** 0.5) / 2

    # Generate x and y coordinates for hexagon centers
    x_coords = np.arange(minx, maxx, hex_width * 3 / 4)
    y_coords_even = np.arange(miny, maxy, hex_height)
    y_coords_odd = y_coords_even + (hex_height / 2)
    hexagons = []
    
    # Generate grid of hexagon centers
    centers = []
    for i, x in enumerate(x_coords):
        y_coords = y_coords_even if i % 2 == 0 else y_coords_odd
        centers.extend([(x, y) for y in y_coords])
    # print("len(centers):", len(centers))

    # Vectorized creation of hexagons
    hexagons = [
        Polygon([
            (x + dx, y + dy)
            for dx, dy in [
                (0, 0),
                (hex_width / 2, hex_height / 2),
                (0, hex_height),
                (-hex_width / 2, hex_height / 2),
                (-hex_width / 2, -hex_height / 2),
                (0, -hex_height),
                (hex_width / 2, -hex_height / 2),
                (0, 0),
            ]
        ])
        for x, y in centers
    ]

    return hexagons

# process the geo data of a given us state
def processState(state):
    
    sizes = [0.01, 0.05, 0.1, 0.5, 1]

    # Load state data file
    print(f"loading data ... state: {state}")
    geofeather_file = f'{data_feather_dir}/us/{state}/{state}.feather'
    data = gpd.read_feather(geofeather_file)
    data['number_int'] = data['number'].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=['number_int'])
    data = data.set_geometry("geometry")

    # print("creating hexagons ...")
    # Create hexagonal grid
    # bounds = data.total_bounds
    bounds = boundsForState(state)

    for hex_size in sizes:
        print(f"processing state: {state}, hex_size: {hex_size}")

        hexagons = create_hex_grid(bounds, hex_size)
        # print("creating hex grid ...")
        hex_grid = gpd.GeoDataFrame(geometry=hexagons, crs=data.crs)

        # print("assigning points to hexagons ...")
        # Spatial join: assign points to hexagons

        joined = gpd.sjoin(data, hex_grid, how="left", predicate="within")

        # print("calculating agg funcs ...")
        # Aggregate statistics
        grouped = joined.groupby("index_right").agg(
            len=("number_int", "count"),
            sum=("number_int", "sum"),
            avg=("number_int", "mean"),
            min=("number_int", "min"),
            max=("number_int", "max"),
            std=("number_int", lambda x: round(pd.Series.std(x), 2)),
            mod=("number_int", lambda x: pd.Series.mode(x).mean())
        ).reset_index()

        # print("cleaning output data ...")

        # Reproject to a projected CRS
        projected_crs = "EPSG:3857"  # Web Mercator
        hex_grid = hex_grid.to_crs(projected_crs)

        # Reset index to create 'index' column for merging
        hex_grid = hex_grid.reset_index(drop=False)

        # Calculate centroids
        hex_grid["geometry"] = hex_grid["geometry"].centroid

        # Reproject back to original CRS
        hex_grid = hex_grid.to_crs(data.crs)

        # Merge back with grouped statistics
        result = hex_grid.merge(grouped, left_on="index", right_on="index_right", how="left")

        # Remove hexbins with no data points
        result = result[result["len"].notna() & (result["len"] > 0)]

        # Remove unnecessary columns
        result = result.drop(columns=["index", "index_right"])

        # Convert selected columns to integers
        columns_to_convert = ["len", "sum", "min", "max", 'avg', 'mod']
        result[columns_to_convert] = result[columns_to_convert].fillna(0).astype(int)

        # Reduce precision of coordinates
        result.geometry = shapely.set_precision(result.geometry, grid_size=0.00001)

        # Make output directories
        output_dir_path = os.path.join(data_out_dir, f"aggregate/{hex_size}/us/{state}")
        os.makedirs(output_dir_path, exist_ok=True)

        # Export to GeoJSON
        output_file_path = f"{output_dir_path}/data.geojson"
        print(f"writing to file ... '{output_file_path}'")
        result.to_file(output_file_path, driver="GeoJSON")

        # Export to NDJSON (each feature on a new line)
        # with open("output.geojson", "w") as f:
        #     for _, row in result.iterrows():
        #         feature = {
        #             "type": "Feature",
        #             "geometry": mapping(row.geometry),
        #             "properties": row.drop("geometry").to_dict()
        #         }
        #         f.write(json.dumps(feature) + "\n")

def processUsCities():
    print("processing us cities ...")
    city_bounds = gpd.read_file('city_boundaries.geojson')
    city_bounds = city_bounds.sort_values('POP2010', ascending=False).head(50)

    for stateIndex, state in enumerate(states):
        # Load state data file
        print(f"[{stateIndex+1}/{len(states)}] loading data ... state: {state}")

        # get cities within the state
        state_cities = city_bounds.loc[city_bounds['ST'] == state.upper()]
        if len(state_cities) == 0:
            continue

        geofeather_file = f'{data_feather_dir}/us/{state}/{state}.feather'
        data = gpd.read_feather(geofeather_file)
        data['number_int'] = data['number'].apply(pd.to_numeric, errors='coerce')
        data.dropna(subset=['number_int'])
        data = data.set_geometry("geometry")

        # match points to each city and filter columns
        joined = gpd.sjoin(data, state_cities, how="left", predicate="within")
        joined = joined.dropna(subset=['index_right'])
        joined = joined[['number_int', 'geometry', 'NAME']]
        joined = joined.rename(columns={'number_int': 'number'})

        # Make output directories
        output_dir_path = os.path.join(data_out_dir, f"us_50_cities/{state}")
        os.makedirs(output_dir_path, exist_ok=True)

        # iterate over every city and save results
        cityNames = joined['NAME'].drop_duplicates()
        for city in cityNames:
            cityNiceName = city.lower().replace(' ', '_')
            cityDf = joined.loc[joined['NAME'] == city][['number', 'geometry']]
            output_file_path = f"{output_dir_path}/{cityNiceName}.geojson"
            print(f"writing to file ... '{output_file_path}'")
            cityDf.to_file(output_file_path, driver="GeoJSON")

            # Convert to GeoJSON format
            geojson_bytes = os.path.getsize(output_file_path)
            
            # split file into 100mb chunks
            if geojson_bytes > 104857600:
                print(f"File exceeds 100MB, splitting...")
                chunk_size = len(cityDf) // (geojson_bytes // 104857600 + 1)
                
                for i, chunk in enumerate(range(0, len(cityDf), chunk_size)):
                    chunk_df = cityDf.iloc[chunk:chunk + chunk_size]
                    chunk_output_path = f"{output_dir_path}/{cityNiceName}_{i+1}.geojson"
                    if len(chunk_df) <= 1:
                        continue
                    print(f"Writing chunk {i+1} to '{chunk_output_path}'")
                    chunk_df.to_file(chunk_output_path, driver="GeoJSON")

                os.remove(output_file_path)


def processUSA():

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(processState, states)

# conmbine and reduce input data files of a given country
def preprocessCountry(country):
    print(f"preprocessing: {country}")

    countryPath = os.path.join(data_in_dir, country)
    states = [f for f in os.listdir(countryPath) if f.isalpha()]
    
    hashes = set()

    counterKeys = ['total', 'used', 'hash_collision', 'invalid', 'out_of_bounds', 'unit_skip', 'empty_num', 'empty_postcode', 'empty_street', 'maxVal']
    counters = {x:0 for x in counterKeys}

    prev_num = ""
    prev_unit = ""
    prev_street = ""

    for stateIndex, state in enumerate(states):

        state_input_path = os.path.join(countryPath, state)
        feather_output_dir = os.path.join(data_feather_dir, os.path.join(country, state))
        os.makedirs(feather_output_dir)
        data_files = [x for x in os.listdir(state_input_path) if os.path.isfile(os.path.join(state_input_path, x)) if "addresses" in x and ".meta" not in x]

        combinedOutputFn = os.path.join(state_input_path, f"temp-{state}.geojson")
        combinedOutputFile = open(combinedOutputFn, 'w')

        for fnIndex, fn in enumerate(data_files):
            print(f"[{stateIndex+1}/{len(states)}][{fnIndex+1}/{len(data_files)}] preprocessing: '{state}/{fn}'")

            with open(state_input_path+'/'+fn, 'r') as f:
                for line in f:
                    counters['total'] += 1
                    data = json.loads(line)

                    hash = data['properties']['hash']
                    if hash in hashes:
                        counters['hash_collision'] += 1
                        continue
                    hashes.add(hash)

                    if data['properties']['number'] == '':
                        counters['empty_num'] += 1
                        continue

                    if data['properties']['number'] == prev_num and data['properties']['street'] == prev_street and data['properties']['unit'] != prev_unit:
                        counters['unit_skip'] += 1
                        continue

                    if data['properties']['postcode'] == '':
                        counters['empty_postcode'] += 1
                        # continue    
                    
                    if data['properties']['street'] == '':
                        counters['empty_street'] += 1
                        # continue

                    prev_num = data['properties']['number']
                    prev_unit = data['properties']['unit']
                    prev_street = data['properties']['street']
                    
                    try:
                        num = int(data['properties']['number'])
                        data['properties']['number'] = num
                    except:
                        counters['invalid'] += 1
                        continue

                    if num <= 0 or num >= 99999:
                        counters['out_of_bounds'] += 1
                        continue

                    if num > counters['maxVal']:
                        counters['maxVal'] = num
                        # print(line)

                    counters['used'] += 1

                    del data['properties']['unit']
                    del data['properties']['hash']
                    del data['properties']['city']
                    del data['properties']['district']
                    del data['properties']['region']
                    del data['properties']['id']
                    del data['properties']['street']
                    del data['properties']['postcode']

                    combinedOutputFile.write(json.dumps(data)+"\n")

        print("rereading combined file ...")
        combinedOutputFile.close()
        gdf = gpd.read_file(combinedOutputFn)
        print(gdf)

        feather_output_path = os.path.join(feather_output_dir, f"{state}.feather")
        print("writing to feather ...", feather_output_path)
        gdf.to_feather(feather_output_path)

        print("deleting combined file ...")
        os.remove(combinedOutputFn)

    counters['total_unused'] = counters['total'] - counters['used']

    print("\nstats:")
    for key in counters:
        rate = (counters[key] / counters['total']) * 100
        print(f'{key} = {counters[key]} ({rate:.2f}%)')

def preprocess(country_dirs_override=None):

    # reset the feather directory
    if os.path.isdir(data_feather_dir):
        shutil.rmtree(data_feather_dir)
    os.mkdir(data_feather_dir)

    # get list of countries to preprocess
    country_dirs = country_dirs_override if country_dirs_override else [f for f in os.listdir(data_in_dir) if f.isalpha()]
    
    # preprocess every listed country
    for country in country_dirs:
        preprocessCountry(country)

if __name__ == '__main__':

    # reset the output directory
    # if os.path.isdir(data_out_dir):
    #     shutil.rmtree(data_out_dir)
    # os.mkdir(data_out_dir)

    # preprocess(['us'])
    # processUSA()
    processUsCities()
