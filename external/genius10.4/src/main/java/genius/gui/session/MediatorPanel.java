package genius.gui.session;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.repository.PartyRepItem;
import genius.gui.panels.ComboboxSelectionPanel;
import genius.gui.panels.LabelAndComponent;
import genius.gui.panels.SingleSelectionModel;
import genius.gui.panels.TextModel;
import genius.gui.panels.TextPanel;
import genius.gui.panels.VflowPanelWithBorder;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * The mediator selector panel for single session. Visible only when the
 * strategy list is non-empty.
 */
@SuppressWarnings("serial")
public class MediatorPanel extends VflowPanelWithBorder {

	private SingleSelectionModel<PartyRepItem> partyModel;

	public MediatorPanel(TextModel nameModel, SingleSelectionModel<PartyRepItem> partyModel) {
		super("Mediator");
		this.partyModel = partyModel;
		final ComboboxSelectionPanel<PartyRepItem> mediatorcomb = new ComboboxSelectionPanel<>("Mediator Strategy",
				partyModel);

		mediatorcomb.setCellRenderer(new RepItemListCellRenderer());

		add(new LabelAndComponent("Mediator ID", new TextPanel(nameModel)));
		add(mediatorcomb);
		updateVisibility();

		connect();
	}

	private void connect() {
		partyModel.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {
				updateVisibility();
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
				updateVisibility();
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				updateVisibility();
			}
		});
	}

	private void updateVisibility() {
		setVisible(!partyModel.getAllItems().isEmpty());
	}

}
