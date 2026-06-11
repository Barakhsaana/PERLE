"""Utilities for parsing mmCIF files into structured Python data and DataFrames."""
import re
import pandas as pd

def remove_annotations(text):
    """
    Remove annotations from a text.
    
    Args:
        text (str): The text to remove comments from.
    
    Returns:
        str: The text with comments removed.
    """
    pattern_full_line_comment = r'^#.*(?:\r?\n|$)'        # Full-line comments (line starting with #, including \n)
    # pattern_end_line_comment = r'#.*$'                  # end-of-line comments (from # to end of line, keep \n)
    text = re.sub(pattern_full_line_comment, '', text, flags=re.MULTILINE)
    # text = re.sub(pattern_end_line_comment, '', text, flags=re.MULTILINE)
    return text



def split_data_blocks(cif_string):
    """
    Split the CIF string into data blocks using "data_xxx" as the delimiter.
    
    Args:
        cif_string (str): The CIF string to split.
    
    Returns:
        list: A list of data blocks.
    """
    pattern_data_block = r"^data_[^\s]*\s*$"
    data_block_splits = re.split(pattern_data_block, cif_string, flags=re.MULTILINE)

    if data_block_splits and data_block_splits[0].strip() == "":
        data_block_splits = data_block_splits[1:]
    else:
        raise AssertionError("First data block is not empty or file does not start with data_")

    return data_block_splits



def split_loop_tags(data_block):
    """
    Split the data block by "loop_"

    Args:
        data_block (str): The data block to split.

    Returns:
        list: A list of loop segmented paragraphs.
    """
    pattern_loop = r"^loop_$\n"
    loop_splits = re.split(pattern_loop, data_block, flags=re.MULTILINE)
    return loop_splits


def mmcif_tokenize_minimal(text: str):
    """
    Minimal mmCIF tokenizer:
    - whitespace (including line breaks) are treated as delimiters
    - support '...' and "..." quoted tokens (internal whitespace allowed)
    - support multi-line text with semicolon: line starting with ';' is the beginning, 
    line ending with ';' is the end, return internal original text, keep line breaks
    - ignore lines starting with # (for safety, although they have already been removed)
    """
    _token_re = re.compile(r"(?:'(?:[^']*)'|\"(?:[^\"]*)\"|[^\s]+)")    # data without gap or quoted by ';' or quoted by ' and "
    lines = text.splitlines(True)  # keep line endings
    in_multi_line = False
    multi_line_buffer = []

    for line in lines:
        if not in_multi_line and line.startswith('#'):
            continue

        if in_multi_line:
            if line.startswith(';'):  # terminator
                yield ''.join(multi_line_buffer)
                multi_line_buffer.clear()
                in_multi_line = False
            else:
                multi_line_buffer.append(line)
            continue

        if line.startswith(';'):  # initiator
            in_multi_line = True
            multi_line_buffer.clear()
            continue

        for tok in _token_re.findall(line):
            if tok and tok[0] in ("'", '"'):
                yield tok[1:-1]
            else:
                yield tok


def load_cif_to_data_dict(cif_file_path):
    """
    Load a CIF file and return a dictionary of data blocks.
    Note: currently only supports single data block
    
    Args:
        cif_file_path (str): The path to the CIF file.
    
    Returns:
        dict: A dictionary of data blocks.
    """


    # load cif file
    with open(cif_file_path, "r", encoding="utf-8") as f:
        cif_string = f.read()
    cif_string = cif_string.strip()
    cif_string = remove_annotations(cif_string).strip()              # remove comments

    data_block_splits = split_data_blocks(cif_string)

    # for data_block in data_block_splits:
    assert len(data_block_splits) == 1, "Only one data block is supported"
    data_block = data_block_splits[0]
    loop_splits = split_loop_tags(data_block)

    data_dict = {}

    for loop_string, id_loop_string in zip(loop_splits, range(len(loop_splits))):
        # loop data is the first non-empty item in tag_splits
        loop_tag_list = []
        loop_value_list = []

        # split tags
        pattern_tag = r"^_[^\s]*"   # start with _ and no space
        tag_matches = re.compile(pattern_tag, re.MULTILINE).findall(loop_string)
        tag_splits = re.split(pattern_tag, loop_string, flags=re.MULTILINE)

        pattern_data = r';.*?;\n|".*?"|\'.*?\'|\S+'      # data without gap or quoted by ';' or quoted by ' and "


        if id_loop_string == 0:
            mode="tag"  # loop or tag
        else:
            mode="loop"  # loop or tag

        for i in range(len(tag_splits)):
            if mode == "loop" and tag_splits[i].strip() == '':
                # collect the tags in the loop if there is no value table
                loop_tag_list.append(tag_matches[i])
            elif mode == "loop" and tag_splits[i].strip() != '':
                # parse the value table
                value_block = tag_splits[i].strip()
                ncol = len(loop_tag_list)

                tokens = list(mmcif_tokenize_minimal(value_block))

                if ncol == 0:
                    raise ValueError("Loop tag list is empty; cannot parse loop values.")

                if len(tokens) % ncol != 0:
                    tail = tokens[-min(30, len(tokens)):]
                    raise ValueError(
                        f"Loop values not aligned: got {len(tokens)} tokens, "
                        f"but ncol={ncol}. Tail tokens: {tail}"
                    )

                for j in range(0, len(tokens), ncol):
                    loop_value_list.append(tokens[j:j+ncol])

                # check length of the loop value list
                for row in loop_value_list:
                    assert len(row) == ncol, (
                        f"Length of one loop row does not match loop tag list: "
                        f"ncol={ncol}, row_len={len(row)}"
                    )

                # write to data_dict
                if len(loop_value_list) != 0:
                    for k in range(ncol):
                        tag = loop_tag_list[k]
                        value_list = [loop_value_list[r][k] for r in range(len(loop_value_list))]
                        if len(value_list) == 1:
                            value_list = value_list[0]
                        data_dict[tag] = value_list

                mode = "tag"
            elif mode == "tag" and tag_splits[i].strip() == '':
                pass
            elif mode == "tag" and tag_splits[i].strip() != '':
                # print(tag_matches[i-1], ":", tag_splits[i].strip())
                data_dict[tag_matches[i-1]] = tag_splits[i].strip()
            else:
                pass


    return data_dict


def get_atom_site_data(data_dict):
    """
    Get atom site data from the data dictionary.
    """
    key_list = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]


    atom_site_df = pd.DataFrame()
    for key in key_list:
        atom_site_df[key] = data_dict[key]

    return atom_site_df



