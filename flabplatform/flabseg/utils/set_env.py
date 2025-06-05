
import datetime
import warnings

from flabplatform.core.registry import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmseg into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmseg default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmseg`, and all registries will build modules from mmseg's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import flabplatform.flabseg.datasets  # noqa: F401,F403
    import flabplatform.flabseg.engine  # noqa: F401,F403
    import flabplatform.flabseg.evaluation  # noqa: F401,F403
    import flabplatform.flabseg.models  # noqa: F401,F403
    import flabplatform.flabseg.structures  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('flabplatform.flabseg')
        if never_created:
            DefaultScope.get_instance('flabplatform.flabseg', scope_name='flabplatform.flabseg')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'flabplatform.flabseg':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "flabplatform.flabseg", '
                          '`register_all_modules` will force the current'
                          'default scope to be "flabplatform.flabseg". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'flabplatform.flabseg-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='flabplatform.flabseg')
