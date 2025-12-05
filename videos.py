import sys
import time
import argparse

import gurobipy as gp
from gurobipy import GRB


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        V, E, R, C, X = map(int, f.readline().split())
        video_sizes = list(map(int, f.readline().split()))
        assert len(video_sizes) == V

        endpoints = []
        for e in range(E):
            dc_latency, K = map(int, f.readline().split())
            cache_latencies = {}
            for _ in range(K):
                c_id, lat = map(int, f.readline().split())
                cache_latencies[c_id] = lat
            endpoints.append({
                "dc_latency": dc_latency,
                "cache_latencies": cache_latencies,
            })

        requests = {}
        for _ in range(R):
            v_id, e_id, n_req = map(int, f.readline().split())
            key = (e_id, v_id)
            requests[key] = requests.get(key, 0) + n_req

    return {
        "V": V, "E": E, "R": R, "C": C, "X": X,
        "video_sizes": video_sizes,
        "endpoints": endpoints,
        "requests": requests,
    }


def build_model(data, mip_gap=5e-3):
    V, E, C, X = data["V"], data["E"], data["C"], data["X"]
    video_sizes = data["video_sizes"]
    endpoints = data["endpoints"]
    requests = data["requests"]

    model = gp.Model("videos")
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1

    videos = range(V)
    caches = range(C)

    print("Création des variables y[v,c]...")
    y = model.addVars(V, C, vtype=GRB.BINARY, name="y")

    print("Pré-calcul des gains et création des variables z[e,v,c] utiles...")
    z_keys = []
    z_gain = {}
    ev_to_caches = {}

    for (e, v), n_req in requests.items():
        ep = endpoints[e]
        dc_lat = ep["dc_latency"]
        for c, lat_ec in ep["cache_latencies"].items():
            save = dc_lat - lat_ec
            if save <= 0:
                continue
            key = (e, v, c)
            z_keys.append(key)
            z_gain[key] = n_req * save
            ev_to_caches.setdefault((e, v), []).append(c)

    print(f"Nombre de couples (e,v) : {len(requests)}")
    print(f"Nombre de variables z : {len(z_keys)}")

    z = model.addVars(z_keys, vtype=GRB.BINARY, name="z")

    print("Contraintes de capacité...")
    for c in caches:
        model.addConstr(
            gp.quicksum(video_sizes[v] * y[v, c] for v in videos) <= X,
            name=f"cap_cache_{c}"
        )

    print("Contraintes z <= y...")
    for (e, v, c) in z_keys:
        model.addConstr(z[e, v, c] <= y[v, c], name=f"link_e{e}_v{v}_c{c}")

    print("Contraintes somme_c z[e,v,c] <= 1...")
    for (e, v), cache_list in ev_to_caches.items():
        model.addConstr(
            gp.quicksum(z[e, v, c] for c in cache_list) <= 1,
            name=f"unique_e{e}_v{v}"
        )

    print("Définition de l'objectif...")
    obj = gp.quicksum(z_gain[k] * z[k] for k in z_keys)
    model.setObjective(obj, GRB.MAXIMIZE)

    print("Modèle construit.")
    print("Variables :", model.NumVars)
    print("Contraintes :", model.NumConstrs)

    return model, y


def write_solution(path_out, y, data):
    V, C = data["V"], data["C"]
    videos = range(V)
    caches = range(C)

    cache_to_videos = {}
    for c in caches:
        vids = [v for v in videos if y[v, c].X > 0.5]
        if vids:
            cache_to_videos[c] = vids

    with open(path_out, "w", encoding="utf-8") as f:
        f.write(str(len(cache_to_videos)) + "\n")
        for c, vids in cache_to_videos.items():
            f.write(str(c) + " " + " ".join(str(v) for v in vids) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="chemin vers le fichier .in")
    args = parser.parse_args()

    dataset_path = args.dataset

    print(f"Lecture du dataset : {dataset_path}")
    t0 = time.time()
    data = read_instance(dataset_path)
    t1 = time.time()
    print(f"Instance lue en {t1 - t0:.3f} s")

    print("Construction du modèle...")
    t2 = time.time()
    model, y = build_model(data, mip_gap=5e-3)
    t3 = time.time()
    print(f"Modèle construit en {t3 - t2:.3f} s")

    print("Écriture du modèle MPS dans videos.mps ...")
    model.write("videos.mps")

    print("Optimisation...")
    t4 = time.time()
    model.optimize()
    t5 = time.time()
    print(f"Optimisation en {t5 - t4:.3f} s")

    if model.SolCount == 0:
        print("Pas de solution trouvée, videos.out non écrit.")
        return

    print("Écriture de la solution dans videos.out ...")
    write_solution("videos.out", y, data)
    print("Terminé.")


if __name__ == "__main__":
    main()
