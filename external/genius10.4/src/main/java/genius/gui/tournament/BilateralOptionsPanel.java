package genius.gui.tournament;

import genius.core.listener.Listener;
import genius.gui.panels.CheckboxPanel;
import genius.gui.panels.VflowPanelWithBorder;

/**
 * Panel that shows options only available when you do bilateral negotiations (2
 * agents per session).
 */
@SuppressWarnings("serial")
public class BilateralOptionsPanel extends VflowPanelWithBorder {

	private BilateralOptionsModel model;
	private PartiesAndProfilesPanel agentsProfilesPanel;

	public BilateralOptionsPanel(BilateralOptionsModel model) {
		super("Special bilateral options");
		this.model = model;
		add(new CheckboxPanel("Agents play both sides", model.getPlayBothSides()));

		agentsProfilesPanel = new PartiesAndProfilesPanel(model.getPartyModelB(), model.getProfileModelB());
		add(agentsProfilesPanel);

		model.getPlayBothSides().addListener(new Listener<Boolean>() {

			@Override
			public void notifyChange(Boolean data) {
				updateVisibility();
			}

		});
		updateVisibility();
	}

	private void updateVisibility() {
		agentsProfilesPanel.setVisible(!model.getPlayBothSides().getValue());
	}

}
