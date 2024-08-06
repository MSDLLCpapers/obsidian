{{ objname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: Module Attributes
      
   .. autosummary::
   {% for item in attributes %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: Functions

   .. autosummary::
      :template: base.rst
      :toctree:
   {% for item in functions %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: Classes

   .. autosummary::
      :template: class.rst
      :toctree:
   {% for item in classes %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :template: base.rst
   {% for item in exceptions %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in modules %}
{% if not item in ("obsidian.acquisition.aq_config", "obsidian.surrogates.surrogate_config", "obsidian.analysis") %}
   {{ item }}
{%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}

