package genius.gui.boaparties;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.boaframework.BOA;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.BoaType;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.RepositoryFactory;
import genius.core.repository.boa.BoaRepItem;
import genius.core.repository.boa.BoaRepItemList;
import genius.core.repository.boa.BoaRepository;
import genius.core.repository.boa.BoaWithSettingsRepItem;
import genius.core.repository.boa.ParameterList;
import genius.core.repository.boa.ParameterRepItem;
import genius.gui.panels.SingleSelectionModel;

/**
 * Contains user settings to create a {@link BoaWithSettingsRepItem}. The user
 * settings differ from the {@link BoaWithSettingsRepItem} because the user can
 * enter ranges for all parameters where the {@link BoaWithSettingsRepItem}
 * contains just 1 setting per parameter.
 *
 * @param <T>
 *            the type of {@link BOA} component that this model is manipulating,
 *            and that comes out of the {@link BoaWithSettingsRepItem} that is
 *            delivered.
 */
public class BoaComponentModel<T extends BOA> {

	/**
	 * List of selectable components of the given type.
	 */
	private SingleSelectionModel<BoaRepItem<T>> componentsListModel;

	private BoaParametersModel parametersModel;

	/**
	 * Creates a model from given existing settings.
	 * 
	 * @param existingItem
	 *            the existing settings. Type must match T.
	 * @throws InstantiateException
	 *             if problem with repo
	 */
	public BoaComponentModel(final BoaWithSettingsRepItem<T> existingItem) {
		BoaType type = existingItem.getBoa().getType();
		loadComponents(type);
		componentsListModel.setSelectedItem(existingItem.getBoa());

		Set<BOAparameter> boaparams = new HashSet<BOAparameter>();
		for (ParameterRepItem e : existingItem.getParameters()) {
			String descr = getDescription(componentsListModel.getSelection(), e.getName());
			boaparams.add(new BOAparameter(e.getName(), e.getValue(), descr));
		}
		parametersModel = new BoaParametersModel(boaparams);
		connect();

	}

	/**
	 * The description is not saved in the RepItem. We find back the original
	 * spec and get desc from there.
	 * 
	 * @param boaRepItem
	 *            a {@link BoaRepItem} that has a parameter with given name
	 * @param name
	 *            a parameter name
	 * @return the original description of the parameter with the given name.
	 */
	private String getDescription(BoaRepItem<T> boaRepItem, String name) {
		try {
			Set<BOAparameter> params = boaRepItem.getInstance().getParameterSpec();
			for (BOAparameter param : params) {
				if (param.getName().equals(name)) {
					return param.getDescription();
				}
			}
			throw new IllegalArgumentException("unknown parameter " + name);
		} catch (Exception e) {
			return "ERR" + e.getMessage();
		}
	}

	/**
	 * Construct model with default settings for given type.
	 * 
	 * @param type
	 *            the type, must match T.
	 */
	public BoaComponentModel(BoaType type) {
		loadComponents(type);
		// select first, so that we always have a proper selection (for
		// #refreshParams)
		componentsListModel.setSelectedItem(componentsListModel.getAllItems().get(0));
		parametersModel = new BoaParametersModel(new HashSet<BOAparameter>());
		resetParams(); // load the default;
		connect();
	}

	/**
	 * Connects listener to ensure {@link #resetParams()} is called when
	 * something changes. Also calls {@link #resetParams()} a first time.
	 */
	private void connect() {

		componentsListModel.addListDataListener(new ListDataListener() {
			@Override
			public void intervalRemoved(ListDataEvent e) {
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				resetParams();
			}
		});

	}

	/**
	 * load all default parameter settings. ASSUMES current selection is valid.
	 */
	private void resetParams() {
		try {
			parametersModel.setParameters(componentsListModel.getSelection().getInstance().getParameterSpec());
		} catch (InstantiateException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Load all available components for given type
	 * 
	 * @param type
	 *            the {@link BoaType} that this model is dealing with. Should
	 *            match T.
	 */
	private void loadComponents(BoaType type) {
		if (type == BoaType.UNKNOWN || type == null) {
			throw new IllegalArgumentException("unsupported type=" + type);
		}
		BoaRepItemList<BoaRepItem<T>> possibleComponents = getBoaRepo().getBoaComponents(type);
		this.componentsListModel = new SingleSelectionModel<BoaRepItem<T>>(possibleComponents);
	}

	/**
	 * Factory method, for testing.
	 * 
	 * @return boa repository
	 */
	protected BoaRepository getBoaRepo() {
		return RepositoryFactory.getBoaRepository();
	}

	/**
	 * all available settings.
	 */
	public List<BoaWithSettingsRepItem<T>> getValues() {
		List<BoaWithSettingsRepItem<T>> list = new ArrayList<>();
		for (ParameterList setting : parametersModel.getSettings()) {
			list.add(new BoaWithSettingsRepItem<T>(componentsListModel.getSelection(), setting));
		}
		return list;
	}

	public SingleSelectionModel<BoaRepItem<T>> getComponentsListModel() {
		return componentsListModel;
	}

	/**
	 * @return the {@link BoaParametersModel}.
	 */
	public BoaParametersModel getParameters() {
		return parametersModel;
	}

}
