def construct_all_predictions(self, model):
    with torch.no_grad():
        all_predictions = None

        # for i, train_data in enumerate(loaders["train_loader"]):
        # for i, train_data in enumerate(loaders["train_loader"]):
        for i, train_data in enumerate(data["train"]):
            sentences, targets = train_data
            features = Variable(sentences)

            if opt.cuda:
                features = features.cuda()

            preds = model(features)
            preds = nn.functional.softmax(preds, dim=1)
            if i == 0:
                all_predictions = preds
            else:
                all_predictions = torch.cat((all_predictions, preds), dim=0)
            del preds
            del features
            del sentences
            del targets

        if "all_predictions" in data:
            del data["all_predictions"]

        data["all_predictions"] = all_predictions.sort(dim=1)[0]
