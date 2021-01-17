class TriangleRegion:

    def __init__(self, p1, p2, p3, id):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.edge_map = {
            (p1, p2): None,
            (p2, p3): None,
            (p3, p1): None,
        }
        self.id = id

    def has_edge(self, p1, p2):
        return (p1, p2) in self.edge_map or (p2, p1) in self.edge_map
    
    def set_contour(self, p1, p2, contour):
        assert self.has_edge(p1, p2), f"{(p1, p2)} edge does not exist. Only have {self.edge_map.keys()}"
        key = (p1, p2)
        if not (p1, p2) in self.edge_map:
            key = (p2, p1)
            contour = contour[::-1]
        self.edge_map[key] = contour

    def get_knots(self, p1, p2):
        assert self.has_edge(p1, p2), f"{(p1, p2)} edge does not exist. Only have {self.edge_map.keys()}"
        key = (p1, p2) if (p1, p2) in self.edge_map else (p2, p1)
        return self.edge_map[key] if self.edge_map[key] else list(key)
    
    def __eq__(self, other):
        return self.id == other.id
