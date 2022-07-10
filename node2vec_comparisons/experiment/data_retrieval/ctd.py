from typing import Dict, List
from downloaders import BaseDownloader
from tqdm.auto import tqdm
import pandas as pd
import numpy as np


def get_ctd_csv_file_header(path: str) -> List[str]:
    """Returns the header of the provided CTD file.

    Parameters
    --------------------
    path: str
        Path to the CTD file to load.

    Raises
    --------------------
    ValueError
        If the path does not contain the substring `CTD`.
    ValueError
        If the header was not found in the provided file.
    """
    if "CTD" not in path:
        raise ValueError(
            "The provided path does not seem to be "
            "a CTD file."
        )

    line_starter = "# Fields:"
    next_line_is_header = False
    with open(path, "r") as f:
        for line in f:
            if next_line_is_header:
                return line.strip("#\n ").split(",")
            if line.startswith(line_starter):
                next_line_is_header = True
            if not line.startswith("#"):
                raise ValueError("Header not found!")


def load_ctd_csv_file(path: str, **kwargs: Dict) -> pd.DataFrame:
    """Return the CTD file at the provided path.

    Parameters
    -----------------------
    path: str
        Path to the CTD file to load.
    **kwargs: Dict
        Dictionary of parameters to forward to reading the CSV
    """
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        dtype=str,
        **kwargs
    )
    new_columns = get_ctd_csv_file_header(path)
    if "usecols" in kwargs:
        new_columns = [
            new_columns[col_index]
            for col_index in kwargs["usecols"]
        ]

    df.columns = new_columns
    return df


def get_column_name(
    candidate_column_names: List[str],
    valid_columns: List[str]
) -> str:
    """Returns the column from the given valid set and haystack.

    Parameters
    ---------------------
    candidate_column_names: List[str]
        The needles to search for.
    valid_columns: List[str]
        The haystack to search into.

    Raises
    ---------------------
    ValueError
        If the column cannot be found.
    """
    for candidate_column_name in candidate_column_names:
        if candidate_column_name in valid_columns:
            return candidate_column_name
    raise ValueError(
        "Column name not found between in {} from {}.".format(
            valid_columns,
            candidate_column_names
        )
    )


def load_ctd_node_list(
    path: str,
    **kwargs: Dict
) -> pd.DataFrame:
    """Returns node list dataframe.

    Parameters
    -----------------------
    path: str
        Path to the CTD file to load.
    **kwargs: Dict
        Dictionary of parameters to forward to reading the CSV
    """
    df = load_ctd_csv_file(path, **kwargs)
    candidate_node_name_columns = [
        "ChemicalName",
        "DiseaseName",
        "AnatomyName",
        "GeneName",
        "PathwayName"
    ]
    node_name_column = get_column_name(
        candidate_node_name_columns,
        df.columns[:3]
    )
    candidate_node_id_columns = [
        "ChemicalID",
        "DiseaseID",
        "AnatomyID",
        "GeneSymbol",
        "PathwayID"
    ]
    node_id_column = get_column_name(
        candidate_node_id_columns,
        df.columns[:3]
    )
    columns = [
        column
        for column in ["Definition", "Synonyms", "ExternalSynonyms", "SlimMappings"]
        if column in df.columns
    ]
    df.fillna("", inplace=True)
    description = df[columns].agg(' '.join, axis=1)
    df = df[[df.columns[0], df.columns[1]]]
    df["description"] = description
    df["node_type"] = path.split("/")[-1].split(".")[0].split("_")[1]
    df.rename(
        columns={
            node_name_column: "node_name",
            node_id_column: "node_id",
        },
        inplace=True
    )
    return df


def extend_node_list_from_edge_list(
    node_list: pd.DataFrame,
    path: str,
    subject_node_type: str,
    object_node_type: str,
    **kwargs
):
    small_edge_list = load_ctd_csv_file(
        path,
        nrows=1
    )
    candidate_node_name_columns = [
        "ChemicalName",
        "DiseaseName",
        "AnatomyName",
        "GeneName",
        "PathwayName",
        "GeneSymbol",
        "GOName",
        "GOTermName",
        "PhenotypeName"
    ]
    candidate_node_id_columns = [
        "ChemicalID",
        "DiseaseID",
        "AnatomyID",
        "GeneSymbol",
        "PathwayID",
        "GOID",
        "GOTermID",
        "PhenotypeID"
    ]
    subject_node_name_column = get_column_name(
        candidate_node_name_columns + [
            col.lower()
            for col in candidate_node_name_columns
        ],
        small_edge_list.columns[:2]
    )
    subject_node_id_column = get_column_name(
        candidate_node_id_columns + [
            col.lower()
            for col in candidate_node_id_columns
        ],
        small_edge_list.columns[:2]
    )
    object_node_name_column = get_column_name(
        candidate_node_name_columns + [
            col.lower()
            for col in candidate_node_name_columns
        ],
        small_edge_list.columns[2:6]
    )
    object_node_id_column = get_column_name(
        candidate_node_id_columns + [
            col.lower()
            for col in candidate_node_id_columns
        ],
        small_edge_list.columns[2:6]
    )
    columns = list(small_edge_list.columns)
    column_number_to_use = sorted([
        columns.index(column)
        for column in {
            subject_node_name_column,
            subject_node_id_column,
            object_node_name_column,
            object_node_id_column
        }
    ])
    edge_list = load_ctd_csv_file(
        path,
        usecols=column_number_to_use,
        **kwargs
    )
    subject_node_list: pd.DataFrame = edge_list[[
        subject_node_name_column, subject_node_id_column]]
    subject_node_list.columns = ["node_name", "node_id"]
    subject_node_list.drop_duplicates("node_id", inplace=True)
    subject_node_list.drop(
        index=subject_node_list.index[
            subject_node_list.node_id.isin(node_list.node_id)
        ],
        inplace=True
    )
    subject_node_list["node_type"] = subject_node_type
    node_list = pd.concat([
        node_list,
        subject_node_list
    ], axis=0)
    node_list.reset_index(drop=True, inplace=True)
    columns = [object_node_name_column, object_node_id_column]
    object_node_list = edge_list[[
        object_node_name_column, object_node_id_column]]
    object_node_list.columns = ["node_name", "node_id"]
    object_node_list.drop_duplicates("node_id", inplace=True)
    object_node_list.drop(
        index=object_node_list.index[
            object_node_list.node_id.isin(node_list.node_id)
        ],
        inplace=True
    )
    object_node_list["node_type"] = object_node_type
    node_list = pd.concat([
        node_list,
        object_node_list
    ], axis=0)
    return node_list


def load_ctd_edge_list(
    path: str,
    **kwargs: Dict
) -> pd.DataFrame:
    small_edge_list = load_ctd_csv_file(
        path,
        nrows=1
    )
    candidate_subject_columns = [
        "ChemicalID",
        "DiseaseID",
        "GeneSymbol",
        "GOID"
    ]
    subject_column_name = get_column_name(
        candidate_subject_columns + [
            col.lower()
            for col in candidate_subject_columns
        ],
        small_edge_list.columns[:2]
    )
    candidate_object_columns = [
        "GeneSymbol",
        "DiseaseID",
        "GOTermID",
        "PathwayID",
        "PhenotypeID"
    ]
    object_column_name = get_column_name(
        candidate_object_columns + [
            col.lower()
            for col in candidate_object_columns
        ],
        small_edge_list.columns[2:6]
    )
    column_names = [
        "GeneForms",
        "Organism",
        "Interaction",
        "InteractionActions",
        "DiseaseName",
        "ChemicalName",
        "chemicalname",
        "GeneSymbol",
        "DirectEvidence",
        "InferenceGeneSymbol",
        "Ontology",
        "GOTermName",
        "GOName",
        "PathwayName",
        "CasRN",
        "InferenceChemicalName",
    ]
    columns = [
        column
        for column in column_names + [
            col.lower()
            for col in column_names
        ]
        if column in small_edge_list.columns
    ]
    column_number_to_use = sorted([
        list(small_edge_list.columns).index(column)
        for column in {
            subject_column_name,
            object_column_name,
            *columns
        }
    ])
    df = load_ctd_csv_file(path, usecols=column_number_to_use, **kwargs)
    df.fillna("", inplace=True)
    description = df[columns].agg(' '.join, axis=1)
    df = df[[subject_column_name, object_column_name]]
    df["description"] = description
    df["edge_type"] = " ".join(path.split(
        "/")[-1].split(".")[0].split("_")[1:])
    df.rename(
        columns={
            subject_column_name: "subject",
            object_column_name: "object",
        },
        inplace=True
    )
    return df


def retrieve_ctd(
    target_node_list_path: str,
    target_edge_list_path: str
):
    """Automatically builds the CTD graph node and edge lists at given paths.

    Parameters
    --------------------------
    target_node_list_path: str
        Path where to store the CTD node list
    target_edge_list_path: str
        Path where to store the CTD edge list
    """
    # Definition of the urls to be downloaded
    node_list_urls = [
        "http://ctdbase.org/reports/CTD_chemicals.csv.gz",
        "http://ctdbase.org/reports/CTD_diseases.csv.gz",
        "http://ctdbase.org/reports/CTD_anatomy.csv.gz",
        "http://ctdbase.org/reports/CTD_genes.csv.gz",
        "http://ctdbase.org/reports/CTD_pathways.csv.gz"
    ]

    edge_list_urls = [
        "http://ctdbase.org/reports/CTD_chem_gene_ixns.csv.gz",
        "http://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz",
        "http://ctdbase.org/reports/CTD_chem_go_enriched.csv.gz",
        "http://ctdbase.org/reports/CTD_chem_pathways_enriched.csv.gz",
        "http://ctdbase.org/reports/CTD_genes_diseases.csv.gz",
        "http://ctdbase.org/reports/CTD_genes_pathways.csv.gz",
        "http://ctdbase.org/reports/CTD_diseases_pathways.csv.gz",
        "http://ctdbase.org/reports/CTD_pheno_term_ixns.csv.gz",
        "http://ctdbase.org/reports/CTD_Phenotype-Disease_biological_process_associations.csv.gz",
        "http://ctdbase.org/reports/CTD_Phenotype-Disease_cellular_component_associations.csv.gz",
        "http://ctdbase.org/reports/CTD_Phenotype-Disease_molecular_function_associations.csv.gz"
    ]

    # Actual download of the necessary files.
    downloader = BaseDownloader()
    downloader.download(node_list_urls)
    downloader.download(edge_list_urls)

    # Normalization of the node types from the edge lists
    # to retrieve the nodes used in the edge lists
    # which are missing from the node lists.

    node_types = {
        "CTD_chem_gene_ixns.csv": ("chemicals", "genes"),
        "CTD_chemicals_diseases.csv": ("chemicals", "diseases"),
        "CTD_chem_go_enriched.csv": ("chemicals", "phenotype"),
        "CTD_chem_pathways_enriched.csv": ("chemicals", "pathways"),
        "CTD_genes_diseases.csv": ("genes", "diseases"),
        "CTD_genes_pathways.csv": ("genes", "pathways"),
        "CTD_diseases_pathways.csv": ("diseases", "pathways"),
        "CTD_pheno_term_ixns.csv": ("chemical", "phenotype"),
        "CTD_Phenotype-Disease_biological_process_associations.csv": ("phenotype", "disease"),
        "CTD_Phenotype-Disease_cellular_component_associations.csv": ("phenotype", "disease"),
        "CTD_Phenotype-Disease_molecular_function_associations.csv": ("phenotype", "disease"),
    }

    for url in tqdm(
        edge_list_urls,
        desc="Extending node list",
        leave=False,
        dynamic_ncols=True
    ):
        edge_list_path = "./Downloads/{}".format(url.split("/")[-1][:-3])
        subject_node_type, object_node_type = node_types[edge_list_path.split(
            "/")[-1]]
        node_list = extend_node_list_from_edge_list(
            node_list=node_list,
            path=edge_list_path,
            subject_node_type=subject_node_type,
            object_node_type=object_node_type,
        )

    # Writing the node list to disk as a TSV document.
    node_list.to_csv(target_node_list_path, sep="\t")

    # Creating the edge list by merging the various edge lists
    edge_list = pd.concat([
        load_ctd_edge_list(
            "./Downloads/{}".format(url.split("/")[-1][:-3]),
        )
        for url in tqdm(
            edge_list_urls,
            desc="Loading edge list",
            leave=False,
            dynamic_ncols=True
        )
    ])

    # Dropping the duplicated edges.
    edge_list.drop_duplicates(["subject", "object", "edge_type"], inplace=True)

    # Writing the edge list to disk as a TSV file
    edge_list.to_csv(target_edge_list_path, sep="\t")
