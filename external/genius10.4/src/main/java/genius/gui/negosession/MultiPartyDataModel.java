package genius.gui.negosession;

import java.util.LinkedHashMap;

import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.listener.Listener;
import genius.gui.progress.DataKey;
import genius.gui.progress.DataKeyTableModel;

/**
 * Tracks the Multiparty tournament and keeps a {@link DataKeyTableModel} up to
 * date. This determines the layout of log file and tables. This can be listened
 * to and shown in a table.
 * 
 * @author W.Pasman
 *
 */
@SuppressWarnings("serial")
public class MultiPartyDataModel extends DataKeyTableModel implements Listener<NegotiationEvent> {

	public MultiPartyDataModel(int numAgents) {
		super(makeDataModel(numAgents));
	}

	/**
	 * create the dataModel. This determines what is logged, the exact order of
	 * the columns, etc. Currently it makes a table with ALL known
	 * {@link DataKey}s.
	 * 
	 * @return datamodel that layouts data.
	 */
	private static LinkedHashMap<DataKey, Integer> makeDataModel(int numAgents) {

		LinkedHashMap<DataKey, Integer> colspec = new LinkedHashMap<DataKey, Integer>();
		for (DataKey key : DataKey.values()) 
		{
			if (key.getMultiple())
				colspec.put(key, numAgents);
			else 
				colspec.put(key, 1);
		}
		return colspec;
	}

	@Override
	public void notifyChange(NegotiationEvent e) {
		if (e instanceof SessionEndedNormallyEvent) {
			SessionEndedNormallyEvent e1 = (SessionEndedNormallyEvent) e;
			addRow(e1.getValues());
		} else if (e instanceof SessionFailedEvent) {
			SessionFailedEvent e1 = (SessionFailedEvent) e;
			addRow(e1.getValues());

		}
	}
}
