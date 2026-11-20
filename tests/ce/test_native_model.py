"""Native forward, complete-head CE, and checkpointed gradient parity."""
import os
import unittest
import torch
from native_fixture import NativeFixture
from modeling_nanochat_ce import load_model


@unittest.skipUnless(os.environ.get('NANOCHAT_SRC'), 'set NANOCHAT_SRC to checkout @ 92d63d4')
class NativeModelTests(NativeFixture, unittest.TestCase):
    def test_native_forward_and_full_vocabulary_gradients(self):
        model = load_model(self.cfg)
        self.assertEqual(set(model.backbone.state_dict()), set(self.reference.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in self.reference.parameters()))
        for name, p in model.backbone.state_dict().items():
            torch.testing.assert_close(p, self.reference.state_dict()[name], rtol=0, atol=0)
        ids = torch.randint(0, len(self.tok), (2, 140))  # crosses the short attention window
        labels = torch.full_like(ids, -100)
        labels[0, 110] = 3
        labels[1, 139] = 4
        logits = self.reference(ids)
        expected_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, len(self.tok)), labels[:, 1:].reshape(-1), ignore_index=-100)
        output = model(ids, labels=labels)
        torch.testing.assert_close(output.logits, logits, rtol=0, atol=0)
        torch.testing.assert_close(output.loss, expected_loss)
        self.assertEqual(output.logits.shape[-1], len(self.tok))
        expected_loss.backward()
        model.gradient_checkpointing_enable()
        model(ids, labels=labels).loss.backward()
        for name, param in model.backbone.named_parameters():
            torch.testing.assert_close(param.grad, dict(self.reference.named_parameters())[name].grad)
        # Ordinary full-vocabulary CE updates non-answer head rows too.
        self.assertGreater(model.lm_head.weight.grad[5].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
