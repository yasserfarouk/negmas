package genius.gui.boaparties;

import java.awt.BorderLayout;
import java.awt.GridLayout;

import javax.swing.JOptionPane;
import javax.swing.JPanel;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOA;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.boa.BoaPartyRepItem;
import genius.core.repository.boa.BoaRepItem;
import genius.core.repository.boa.BoaWithSettingsRepItem;
import genius.core.repository.boa.ParameterList;
import genius.gui.panels.LabelAndComponent;
import genius.gui.panels.TextPanel;

/**
 * panel that allows user to set properties that will later be used to generate
 * {@link BoaPartyRepItem}s.
 */
@SuppressWarnings("serial")
public class BoaPartyPanel extends JPanel {

	/**
	 * show panel that edits the given BoaPartyModel
	 * 
	 * @param model
	 *            the {@link BoaPartyModel} to edit.
	 */
	public BoaPartyPanel(BoaPartyModel model) {
		setLayout(new BorderLayout());

		add(new LabelAndComponent("Name", new TextPanel(model.getNameModel())), BorderLayout.NORTH);

		JPanel strategies = new JPanel(new GridLayout(2, 2, 40, 50));
		strategies.add(new BoaComponentPanel<>(model.getOfferingModel(), "Bidding Strategy"));
		strategies.add(new BoaComponentPanel<>(model.getAcceptanceModel(), "Acceptance Strategy"));
		strategies.add(new BoaComponentPanel<>(model.getOpponentModel(), "Opponent Model"));
		strategies.add(new BoaComponentPanel<>(model.getOmStrategiesModel(), "Opponent Model Strategy"));
		add(strategies, BorderLayout.CENTER);

	}

	/**
	 * Test function.
	 * 
	 * @param args
	 * @throws InstantiateException
	 */
	public static void main(String[] args) throws InstantiateException {

		// make specific settings, to check the proper value is selected
		BoaWithSettingsRepItem<OfferingStrategy> boa1 = makeSettings("resources.boa.Offering2");
		BoaWithSettingsRepItem<AcceptanceStrategy> boa2 = makeSettings("resources.boa.Acceptance2");
		BoaWithSettingsRepItem<OpponentModel> boa3 = makeSettings("resources.boa.OpponentModel1");
		BoaWithSettingsRepItem<OMStrategy> boa4 = makeSettings("resources.boa.OMStrategy1");
		BoaPartyRepItem partyitem = new BoaPartyRepItem("test party", boa1, boa2, boa3, boa4);
		BoaPartyModel model = new BoaPartyModel(partyitem);

		int result = JOptionPane.showConfirmDialog(null, new BoaPartyPanel(model), "test model",
				JOptionPane.OK_CANCEL_OPTION);
		System.out.println("result=" + result + ":" + model);
	}

	// for testing
	private static <T extends BOA> BoaWithSettingsRepItem<T> makeSettings(String classname) {
		BoaRepItem<T> item = new BoaRepItem<T>(classname);
		ParameterList parameters = new ParameterList();
		return new BoaWithSettingsRepItem<T>(item, parameters);
	}

}
