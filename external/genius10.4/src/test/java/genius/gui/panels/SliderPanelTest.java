package genius.gui.panels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import javax.swing.JFrame;

import org.junit.Test;

import genius.core.listener.Listener;

public class SliderPanelTest {
	@Test
	public void testPanel() throws InterruptedException {
		IntegerModel model = new IntegerModel(3, 1, 10, 1);
		SliderPanel panel = new SliderPanel("test", model);

		JFrame frame = new JFrame();
		frame.setContentPane(panel);
		frame.pack();
		frame.setVisible(true);
		Thread.sleep(1000);
		assertEquals((Integer) 3, model.getValue());

		// drag the slider
		panel.getSlider().setValue(2);
		assertEquals((Integer) 2, model.getValue());

	}

	private boolean calledback = false;

	@Test
	public void testPanelEnableDisable() throws InterruptedException {
		IntegerModel model = new IntegerModel(3, 1, 10, 1);
		model.addListener(new Listener<Integer>() {

			@Override
			public void notifyChange(Integer data) {
				calledback = true;
			}
		});
		SliderPanel panel = new SliderPanel("test", model);

		JFrame frame = new JFrame();
		frame.setContentPane(panel);
		frame.pack();
		frame.setVisible(true);
		Thread.sleep(1000);

		assertFalse(calledback);
		// drag the slider
		panel.getSlider().setValue(2);
		Thread.sleep(1000);
		assertEquals((Integer) 2, model.getValue());
		assertTrue(calledback);

		calledback = false;
		model.setLock(true);
		Thread.sleep(1000);
		assertTrue(calledback);
		assertFalse(panel.isEnabled());

		calledback = false;
		model.setLock(false);
		Thread.sleep(1000);
		assertTrue(calledback);
		assertTrue(panel.isEnabled());

	}
}
