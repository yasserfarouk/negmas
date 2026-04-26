package genius.gui.session;

import genius.gui.panels.BooleanModel;
import genius.gui.panels.CheckboxPanel;
import genius.gui.panels.VflowPanelWithBorder;

@SuppressWarnings("serial")
public class BilateralOptionsPanel extends VflowPanelWithBorder {

	public BilateralOptionsPanel(BooleanModel utilutilplotModel, BooleanModel showallBidsModel) {
		super("Bilateral options");

		add(new CheckboxPanel("Show Util-Util Graph", utilutilplotModel));
		add(new CheckboxPanel("Show all bids", showallBidsModel));
	}

}
