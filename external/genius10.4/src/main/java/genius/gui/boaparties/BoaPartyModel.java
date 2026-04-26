package genius.gui.boaparties;

import java.util.ArrayList;
import java.util.List;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.exceptions.InstantiateException;
import genius.core.list.ImmutableList;
import genius.core.repository.RepositoryFactory;
import genius.core.repository.boa.BoaPartyRepItem;
import genius.core.repository.boa.BoaRepository;
import genius.core.repository.boa.BoaWithSettingsRepItem;
import genius.gui.panels.TextModel;

/**
 * Stores the settings to create a new {@link BoaPartyRepItem} for editing in a
 * GUI.
 */
public class BoaPartyModel {
	private TextModel nameModel;

	private BoaComponentModel<OfferingStrategy> offeringModel;

	private BoaComponentModel<AcceptanceStrategy> acceptanceModel;

	private BoaComponentModel<OpponentModel> opponentModel;

	private BoaComponentModel<OMStrategy> omStrategiesModel;

	/**
	 * Create Boa Party model with an initial value from an existingItem
	 * 
	 * @param existingItem
	 *            an existing {@link BoaPartyRepItem}
	 * @throws InstantiateException
	 *             if problem with repository
	 */
	public BoaPartyModel(BoaPartyRepItem existingItem) throws InstantiateException {
		nameModel = new TextModel(existingItem.getName());
		initModels(existingItem);
	}

	/**
	 * Load the models with the available values from the repos.
	 * 
	 * @throws InstantiateException
	 *             if repository does not have some type of BOA component
	 * 
	 */
	private void initModels(BoaPartyRepItem party) throws InstantiateException {
		BoaRepository repo = RepositoryFactory.getBoaRepository();
		if (repo.getAcceptanceConditions().isEmpty() || repo.getBiddingStrategies().isEmpty()
				|| repo.getOpponentModels().isEmpty() || repo.getOpponentModelStrategies().isEmpty()) {
			throw new InstantiateException(
					"boarepository is invalid. editor requires at least one valid component of each type in the repository");
		}
		offeringModel = new BoaComponentModel<OfferingStrategy>(party.getOfferingStrategy());
		acceptanceModel = new BoaComponentModel<AcceptanceStrategy>(party.getAcceptanceStrategy());
		opponentModel = new BoaComponentModel<OpponentModel>(party.getOpponentModel());
		omStrategiesModel = new BoaComponentModel<OMStrategy>(party.getOmStrategy());
	}

	public TextModel getNameModel() {
		return nameModel;
	}

	public BoaComponentModel<OfferingStrategy> getOfferingModel() {
		return offeringModel;
	}

	public BoaComponentModel<AcceptanceStrategy> getAcceptanceModel() {
		return acceptanceModel;
	}

	public BoaComponentModel<OpponentModel> getOpponentModel() {
		return opponentModel;
	}

	public BoaComponentModel<OMStrategy> getOmStrategiesModel() {
		return omStrategiesModel;
	}

	@Override
	public String toString() {
		return "BOA:" + nameModel.getText();
	}

	/**
	 * 
	 * @return list of all boacomponent-plus-setting combinations specified by
	 *         this model. FIXME prevent generation of excessive lists. Use
	 *         {@link ImmutableList}
	 */
	public List<BoaPartyRepItem> getValues() {
		List<BoaPartyRepItem> items = new ArrayList<>();
		int serial = 0;
		String name = nameModel.getText();

		for (BoaWithSettingsRepItem<OfferingStrategy> value1 : offeringModel.getValues()) {
			for (BoaWithSettingsRepItem<AcceptanceStrategy> value2 : acceptanceModel.getValues()) {
				for (BoaWithSettingsRepItem<OpponentModel> value3 : opponentModel.getValues()) {
					for (BoaWithSettingsRepItem<OMStrategy> value4 : omStrategiesModel.getValues()) {
						items.add(new BoaPartyRepItem(name + (serial == 0 ? "" : serial), value1, value2, value3,
								value4));
						serial++;
					}
				}
			}
		}

		return items;
	}

}
