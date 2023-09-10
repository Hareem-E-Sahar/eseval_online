import csv 
class CSVData:
    def __init__(self, fix, ns, nd, nf, entrophy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp, contains_bug, author_date_unix_timestamp, commit_type, commit_id):
        self.fix = fix
        self.ns = ns
        self.nd = nd
        self.nf = nf
        self.entrophy = entrophy
        self.la = la
        self.ld = ld
        self.lt = lt
        self.ndev = ndev
        self.age = age
        self.nuc = nuc
        self.exp = exp
        self.rexp = rexp
        self.sexp = sexp
        self.contains_bug = contains_bug
        self.author_date_unix_timestamp = author_date_unix_timestamp
        self.commit_type = commit_type
        self.commit_id = commit_id

    def __eq__(self, other):
        if isinstance(other, CSVData):
            # Compare all attributes except commit_id, commit_type, timestamp for equality
            return (
                self.fix == other.fix
                and self.ns == other.ns
                and self.nd == other.nd
                and self.nf == other.nf
                and self.entrophy == other.entrophy
                and self.la == other.la
                and self.ld == other.ld
                and self.lt == other.lt
                and self.ndev == other.ndev
                and self.age == other.age
                and self.nuc == other.nuc
                and self.exp == other.exp
                and self.rexp == other.rexp
                and self.sexp == other.sexp
                and self.contains_bug == other.contains_bug
                
            )
        return False

def read_commits_csv(csv_file_path):
    csv_data_list = []
    type3_list = []
    with open(csv_file_path, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)  # Assumes the first row contains column headers
        for row in csv_reader:
            csv_data = CSVData(
                fix=str(row['fix']),
                ns=int(row['ns']),
                nd=int(row['nd']),
                nf=int(row['nf']),
                entrophy=float(row['entrophy']),
                la=int(row['la']),
                ld=int(row['ld']),
                lt=float(row['lt']),
                ndev=int(row['ndev']),
                age=float(row['age']),
                nuc=int(row['nuc']),
                exp=float(row['exp']),
                rexp=float(row['rexp']),
                sexp=int(row['sexp']),
                contains_bug=str(row['contains_bug']),
                author_date_unix_timestamp=int(row['author_date_unix_timestamp']),
                commit_type=int(row['commit_type']),
                commit_id=str(row['commit_id'])
            )

            if len(row['commit_id']) > 0:
                csv_data_list.append(csv_data)
            else:
                type3_list.append(csv_data)
    return csv_data_list,type3_list


# Create an empty list to store CSVData objects

'''
data1 = CSVData(False, 2, 79, 976, 8.937691918, 264772, 0, 0, 1, 0, 0, 487.5, 0, 974, False, 1143467626, 0, "a84fabcbc6fee8a69253ad92a304b4718e96a7c9")
data2 = CSVData(False, 2, 79, 976, 8.937691918, 264772, 0, 0, 1, 0, 0, 487.5, 0, 974, False, 1143467626, 0, "")
if data1 == data2:
    print("Objects are equal (excluding commit_id)")
else:
    print("Objects are not equal (excluding commit_id)")

'''
