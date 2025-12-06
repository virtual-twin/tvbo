# -*- coding: utf-8 -*-
"""Lightweight literature add-on for linking PubMed metadata to SimulationStudy records."""

from odoo import api, fields, models


class MeshTerm(models.Model):
    _name = "tvbo.mesh_term"
    _description = "PubMed MeSH term"
    _rec_name = "name"

    name = fields.Char(required=True, index=True)
    tree_number = fields.Char(string="Tree number")
    description = fields.Text()


class LiteratureReference(models.Model):
    _name = "tvbo.literature_reference"
    _description = "Bibliographic reference with PubMed metadata"
    _rec_name = "title"

    title = fields.Char(required=True)
    key = fields.Char(index=True, string="BibTeX key")
    pubmed_id = fields.Char(index=True, string="PubMed ID")
    doi = fields.Char(index=True)
    year = fields.Integer()
    journal = fields.Char()
    abstract = fields.Text()
    authors = fields.Text(string="Author list")
    mesh_term_ids = fields.Many2many(
        comodel_name="tvbo.mesh_term",
        relation="tvbo_literature_mesh_term_rel",
        string="MeSH terms",
    )
    simulation_study_ids = fields.Many2many(
        comodel_name="tvbo.simulation_study",
        relation="tvbo_simulation_study_reference_rel",
        column1="reference_id",
        column2="simulation_study_id",
        string="Simulation studies",
    )
    url = fields.Char(compute="_compute_urls", store=True)
    pubmed_url = fields.Char(compute="_compute_urls", store=True)

    @api.depends("pubmed_id", "doi")
    def _compute_urls(self):
        for rec in self:
            rec.pubmed_url = (
                f"https://pubmed.ncbi.nlm.nih.gov/{rec.pubmed_id}/"
                if rec.pubmed_id
                else False
            )
            if rec.doi:
                rec.url = f"https://doi.org/{rec.doi}"
            elif rec.pubmed_url:
                rec.url = rec.pubmed_url
            else:
                rec.url = False


class SimulationStudy(models.Model):
    _inherit = "tvbo.simulation_study"

    reference_ids = fields.Many2many(
        comodel_name="tvbo.literature_reference",
        relation="tvbo_simulation_study_reference_rel",
        column1="simulation_study_id",
        column2="reference_id",
        string="References",
    )
    reference_count = fields.Integer(
        compute="_compute_reference_count", string="Reference count"
    )

    def _compute_reference_count(self):
        for rec in self:
            rec.reference_count = len(rec.reference_ids)
