package genius.gui.boaparties;

import java.util.ArrayList;
import java.util.List;

import javax.swing.event.TableModelListener;
import javax.swing.table.TableModel;

import genius.core.boaframework.BOA;
import genius.core.boaframework.BoaType;
import genius.core.listener.Listener;
import genius.core.repository.BoaPartyRepository;
import genius.core.repository.RepositoryFactory;
import genius.core.repository.boa.BoaPartyRepItem;
import genius.core.repository.boa.BoaWithSettingsRepItem;

/**
 * Adapts the BoaPartyRepository to a tablemodel
 *
 */
public class BoaPartiesModel implements TableModel {
	private BoaPartyRepository repo;
	private List<TableModelListener> listeners = new ArrayList<>();

	private final static String columns[] = { "Name", BoaType.BIDDINGSTRATEGY.toString(),
			BoaType.ACCEPTANCESTRATEGY.toString(), BoaType.OPPONENTMODEL.toString(), BoaType.OMSTRATEGY.toString() };

	public BoaPartiesModel() {
		repo = RepositoryFactory.getBoaPartyRepository();
		repo.addListener(new Listener<BoaPartyRepItem>() {
			@Override
			public void notifyChange(BoaPartyRepItem data) {
				for (TableModelListener listener : listeners) {
					listener.tableChanged(null); // CHECK is this ok?
				}
			}
		});
	}

	@Override
	public int getRowCount() {
		return repo.getList().getList().size();
	}

	@Override
	public int getColumnCount() {
		return columns.length;
	}

	@Override
	public String getColumnName(int columnIndex) {
		return columns[columnIndex];
	}

	@Override
	public Class<?> getColumnClass(int columnIndex) {
		return String.class;
	}

	@Override
	public boolean isCellEditable(int rowIndex, int columnIndex) {
		return false;
	}

	@Override
	public Object getValueAt(int rowIndex, int columnIndex) {
		BoaPartyRepItem boaparty = repo.getList().getList().get(rowIndex);
		switch (columnIndex) {
		case 0:
			return boaparty.getName();
		case 1:
			return strategyString(boaparty, BoaType.BIDDINGSTRATEGY);
		case 2:
			return strategyString(boaparty, BoaType.ACCEPTANCESTRATEGY);
		case 3:
			return strategyString(boaparty, BoaType.OPPONENTMODEL);
		case 4:
			return strategyString(boaparty, BoaType.OMSTRATEGY);
		}
		return "-";
	}

	private String strategyString(BoaPartyRepItem boaparty, BoaType strategyType) {
		BoaWithSettingsRepItem<? extends BOA> strategy = boaparty.getStrategy(strategyType);
		return strategy.getBoa().getName() + strategy.getParameters();
	}

	@Override
	public void setValueAt(Object aValue, int rowIndex, int columnIndex) {

	}

	@Override
	public void addTableModelListener(TableModelListener l) {
		listeners.add(l);
	}

	@Override
	public void removeTableModelListener(TableModelListener l) {
		listeners.remove(l);

	}
}
