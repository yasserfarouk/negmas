package genius.gui.tournament;

import java.awt.GridLayout;

import javax.swing.JPanel;
import javax.swing.ListCellRenderer;

import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.gui.panels.SubsetSelectionModel;
import genius.gui.panels.SubsetSelectionPanelPlus;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * Panel that shows agents and profiles panels side by side. Both panels get
 * half of the width.
 */
@SuppressWarnings("serial")
public class PartiesAndProfilesPanel extends JPanel {

	@SuppressWarnings("unchecked")
	public PartiesAndProfilesPanel(SubsetSelectionModel<ParticipantRepItem> partyModel,
			SubsetSelectionModel<ProfileRepItem> profileModel) {
		setLayout(new GridLayout(1, 2));

		SubsetSelectionPanelPlus<ParticipantRepItem> leftPanel = new SubsetSelectionPanelPlus<>("Parties", partyModel);
		// ugly cast, RepItemListCellRenderer has incorrect typing...
		leftPanel.setCellRenderer(
				(ListCellRenderer<ParticipantRepItem>) (ListCellRenderer) new RepItemListCellRenderer());
		SubsetSelectionPanelPlus<ProfileRepItem> rightPanel = new SubsetSelectionPanelPlus<>("Profiles", profileModel);
		rightPanel.setCellRenderer((ListCellRenderer<ProfileRepItem>) (ListCellRenderer) new RepItemListCellRenderer());
		add(leftPanel);
		add(rightPanel);
	}

}
