# -*- coding: utf-8 -*-
{
    'name': 'TVBO - The Virtual Brain Ontology',
    'version': '19.0.1.0.0',
    'category': 'Research/Neuroscience',
    'summary': 'Data models for brain simulation, DBS, and neuroanatomical atlases',
    'description': '''
        Complete TVBO data models including:
        - Brain simulation dynamics and experiments
        - Deep Brain Stimulation (DBS) protocols
        - Neuroanatomical atlases and parcellations
    ''',
    'author': 'Charité Universitätsmedizin Berlin',
    'website': 'https://github.com/virtual-twin/tvbo',
    'license': 'EUPL-1.2',
    'depends': ['base', 'website'],
    'data': [
        'security/ir.model.access.csv',
        'views/menus.xml',
        'views/database_views.xml',
        'views/website_templates.xml',
        'data/neural_models.xml',
        'data/integrators.xml',
    ],
    'installable': True,
    'application': True,
    'auto_install': False,
}
