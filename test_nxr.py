import networkxr as nxr

G = nxr.DiGraph()
G.add_edge("A", "B", weight=1.0)
print(G.has_edge("A", "B"))
print(G["A"]["B"]["weight"])
print(G.get_edge_data("A", "B").get("weight"))
