package genius.gui.tree;

import genius.gui.panels.CheckboxPanel;
import genius.gui.panels.DoublePanel;
import genius.gui.panels.SliderPanel;
import genius.gui.panels.VflowPanelWithBorder;

@SuppressWarnings("serial")
public class UncertaintySettingsPanel extends VflowPanelWithBorder {
	public UncertaintySettingsPanel(UncertaintySettingsModel model) {
		super("Uncertainty settings");
		add(new CheckboxPanel("Enable preference uncertainty", model.getIsEnabled()));

		add(new SliderPanel("Number of outcomes in ranking (maximum = " + model.getTotalBids() +"):  ", model.getComparisons()));
		
		// Removed the errors for now but with all functionality intact; uncomment to add back in
		// add(new SliderPanel("Nr. of errors    ", model.getErrors()));	
		
		add(new DoublePanel("Elicitation cost: ", model.getElicitationCost()));		
		
		add(new CheckboxPanel(
				"Fixed seed (for reproducible results)",
				model.getIsFixedSeed()));
		add(new CheckboxPanel(
				"Grant parties access to real utility functions (experimental setup)",
				model.getIsExperimental()));
	}
}
