package genius.gui.boaparties;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.junit.Test;

import genius.core.boaframework.BOAparameter;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.RepItem;
import genius.gui.panels.SingleSelectionModel;

/**
 * bit hacky test. Just opens the panel with a mocked component.
 */
public class BoaComponentPanelTest {
	@SuppressWarnings("rawtypes")
	@Test
	public void testOpenPanel() throws InstantiateException, InterruptedException {
		JFrame frame = new JFrame();
		frame.setLayout(new BorderLayout());

		// Mock a Boa component model to test the GUI
		@SuppressWarnings("rawtypes")
		BoaComponentModel model = mock(BoaComponentModel.class);
		@SuppressWarnings("rawtypes")
		SingleSelectionModel componentlist = mock(SingleSelectionModel.class);
		when(model.getComponentsListModel()).thenReturn(componentlist);
		BoaParametersModel params = mock(BoaParametersModel.class);
		when(model.getParameters()).thenReturn(params);
		when(params.getSetting()).thenReturn(new ArrayList<BOAparameter>());

		RepItem comp1 = mock(RepItem.class);
		when(comp1.toString()).thenReturn("comp1");
		RepItem comp2 = mock(RepItem.class);
		when(comp2.toString()).thenReturn("comp2");
		RepItem comp3 = mock(RepItem.class);
		when(comp3.toString()).thenReturn("comp3");

		List<RepItem> allitems = Arrays.asList(new RepItem[] { comp1, comp2, comp3 });
		when(componentlist.getAllItems()).thenReturn(allitems);
		when(componentlist.getSelectedItem()).thenReturn(allitems.get(0));

		frame.getContentPane().add(new BoaComponentPanel(model, "test boa component panel"), BorderLayout.CENTER);
		frame.pack();
		frame.setVisible(true);
		Thread.sleep(2000);
		frame.setVisible(false);
	}

}
