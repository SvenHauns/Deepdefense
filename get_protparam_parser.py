from Bio.SeqUtils.ProtParam import ProteinAnalysis


def parser(sample):
    try:

        analyse = ProteinAnalysis(str(sample).replace('U', 'S'))
        aa_dict = analyse.count_amino_acids()
        num_amino_acids = '%d' % (sum(list(aa_dict.values())))
        molecular_weight = '%.3f' % (analyse.molecular_weight())
        pI = '%.3f' % (analyse.isoelectric_point())
        neg_charged_residues = '%d' % (aa_dict['D'] + aa_dict['E'])
        pos_charged_residues = '%d' % (aa_dict['K'] + aa_dict['R'])
        extinction_coefficients_1 = '%d' % (aa_dict['Y'] * 1490 + aa_dict['W'] * 5500)
        extinction_coefficients_2 = '%d' % (aa_dict['Y'] * 1490 + aa_dict['W'] * 5500 + aa_dict['C'] * 125)
        instability_index = '%.3f' % (analyse.instability_index())
        gravy = '%.3f' % (analyse.gravy())
        secondary_structure_fraction = tuple(['%.3f' % frac for frac in analyse.secondary_structure_fraction()])
        secondary_structure_fraction = [float(sec) for sec in secondary_structure_fraction]

        analysis = (float(num_amino_acids), float(molecular_weight), float(pI),
                    float(neg_charged_residues), float(pos_charged_residues),
                    float(extinction_coefficients_1), float(extinction_coefficients_2),
                    float(instability_index), float(gravy),
                    *secondary_structure_fraction)

    except:
        analysis = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    return analysis
